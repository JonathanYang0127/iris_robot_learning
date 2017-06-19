"""
For the heatmap, I index into the Q function with Q[state, action]
"""
from collections import namedtuple

import os
import argparse

import itertools
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.autograd import Variable
from pathlib import Path
import re
from operator import itemgetter

from railrl.envs.pygame.water_maze import WaterMaze

HeatMap = namedtuple("HeatMap", ['values', 'state_values', 'action_values'])


def make_heat_map(eval_func, state_bounds, action_bounds, *, resolution=10):
    state_values = np.linspace(*state_bounds, num=resolution)
    action_values = np.linspace(*action_bounds, num=resolution)
    map = np.zeros((resolution, resolution))

    for i in range(resolution):
        for j in range(resolution):
            map[i, j] = eval_func(state_values[i], action_values[j])
    return HeatMap(map, state_values, action_values)


def create_figure(estimated_heatmaps, optimal_heatmaps, target_poses,
                  iteration_number):
    width = 5 * len(estimated_heatmaps)
    height = 20
    fig, axes = plt.subplots(4, len(estimated_heatmaps), figsize=(width, height))
    vmax = max(itertools.chain(
        [hm.values.max() for hm in estimated_heatmaps],
        [hm.values.max() for hm in optimal_heatmaps],
    ))
    vmin = min(itertools.chain(
        [hm.values.min() for hm in estimated_heatmaps],
        [hm.values.min() for hm in optimal_heatmaps],
    ))
    for i, (estimated_heatmap, opt_heatmap, target_pos) in enumerate(
            zip(estimated_heatmaps, optimal_heatmaps, target_poses)
    ):
        min_pos = max(target_pos - WaterMaze.TARGET_RADIUS,
                      -WaterMaze.BOUNDARY_DIST)
        max_pos = min(target_pos + WaterMaze.TARGET_RADIUS,
                      WaterMaze.BOUNDARY_DIST)
        state_values = estimated_heatmap.state_values
        target_right_of = min_pos <= state_values
        target_left_of = state_values <= max_pos
        first_index_on = np.where(target_right_of)[0][0]
        last_index_on = np.where(target_left_of)[0][-1] + 1

        """
        Plot Estimated & Optimal QF
        """
        ax = axes[0][i]
        sns.heatmap(
            estimated_heatmap.values,
            ax=ax,
            xticklabels=False,
            yticklabels=False,
            vmax=vmax,
            vmin=vmin,
        )
        ax.vlines([first_index_on, last_index_on], *ax.get_ylim())
        ax.set_xlabel("Position")
        ax.set_ylabel("Velocity")
        ax.set_title("Estimated QF. Target Position = {0}".format(target_pos))

        ax = axes[1][i]
        sns.heatmap(
            opt_heatmap.values,
            ax=ax,
            xticklabels=False,
            yticklabels=False,
            vmax=vmax,
            vmin=vmin,
        )
        ax.vlines([first_index_on, last_index_on], *ax.get_ylim())
        ax.set_xlabel("Position")
        ax.set_ylabel("Velocity")
        ax.set_title("Optimal QF. Target Position = {0}".format(target_pos))

        """
        Plot Estimated & Optimal VF
        """
        ax = axes[2][i]
        ax.plot(state_values, np.max(estimated_heatmap.values, axis=1))
        ax.vlines([min_pos, max_pos], *ax.get_ylim())
        ax.set_xlabel("Position")
        ax.set_ylabel("Value Function")
        ax.set_title("Estimated VF. Target Position = {0}".format(target_pos))

        ax = axes[3][i]
        ax.plot(state_values, np.max(opt_heatmap.values, axis=1))
        ax.vlines([min_pos, max_pos], *ax.get_ylim())
        ax.set_xlabel("Position")
        ax.set_ylabel("Value Function")
        ax.set_title("Optimal VF. Target Position = {0}".format(target_pos))

    fig.suptitle("Iteration = {0}".format(iteration_number))
    return fig


def create_eval_fnct(qf, target_pos):
    def evaluate(x_pos, x_vel):
        dist = np.linalg.norm(x_pos - target_pos)
        on_target = dist <= WaterMaze.TARGET_RADIUS
        state = np.hstack([x_pos, on_target, target_pos])
        state = Variable(torch.from_numpy(state)).float().unsqueeze(0)

        action = np.array([x_vel])
        action = Variable(torch.from_numpy(action)).float().unsqueeze(0)
        out = qf(state, action)
        return out.data.numpy()
    return evaluate


def clip(value, min_value, max_value):
    return max(min(value, max_value), min_value)


def find_nearest_idx(array, value):
    return (np.abs(array - value)).argmin()


def create_optimal_qf(target_pos, state_bounds, action_bounds, discount_factor,
                      *, resolution=10):
    """
    Do Q-learning to find the optimal Q-values
    :param target_pos: 
    :param state_bounds: 
    :param action_bounds: 
    :param resolution: 
    :return: 
    """
    def get_reward(state):
        return int(target_pos - WaterMaze.TARGET_RADIUS
                   <= state
                   <= target_pos + WaterMaze.TARGET_RADIUS)

    qf = np.zeros((resolution, resolution))  # state, action
    state_values = np.linspace(*state_bounds, num=resolution)
    action_values = np.linspace(*action_bounds, num=resolution)
    alpha = 0.1
    for _ in range(1000):
        vf = np.max(qf, axis=1)
        for action_i, state_i in itertools.product(range(resolution),
                                                   range(resolution)):
            next_state = clip(
                action_values[action_i] + state_values[state_i],
                *state_bounds
            )
            next_state_i = find_nearest_idx(state_values, next_state)
            reward = get_reward(state_values[next_state_i])
            qf[state_i, action_i] = (
                (1 - alpha) * qf[state_i, action_i]
                + alpha * (
                    reward + discount_factor * vf[next_state_i]
                )
            )

    def qf_fnct(state, action):
        state_i = find_nearest_idx(state_values, state)
        action_i = find_nearest_idx(action_values, action)
        return qf[state_i, action_i]

    return qf_fnct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder_path", type=str)
    args = parser.parse_args()
    base = Path(os.getcwd())
    base = base / args.folder_path
    path_and_iter = []
    for pkl_path in base.glob('*.pkl'):
        match = re.search('_(-*[0-9]*).pkl$', str(pkl_path))
        iter_number = int(match.group(1))
        path_and_iter.append((pkl_path, iter_number))
    path_and_iter = sorted(path_and_iter, key=itemgetter(1))

    save_dir = base / "images"
    if not save_dir.exists():
        save_dir.mkdir()

    resolution = 10
    discount_factor = 0.5
    state_bounds = (-WaterMaze.BOUNDARY_DIST, WaterMaze.BOUNDARY_DIST)
    action_bounds = (-1, 1)

    for path, iter_number in path_and_iter:
        data = joblib.load(str(path))
        save_file = save_dir / 'iter_{}.png'.format(iter_number)
        qf = data['qf']
        print("QF loaded from iteration %d" % iter_number)

        heatmaps = []
        optimal_heatmaps = []
        target_poses = np.linspace(-5, 5, num=5)
        for target_pos in target_poses:
            qf_eval = create_eval_fnct(qf, target_pos)
            heatmaps.append(make_heat_map(
                qf_eval,
                state_bounds=state_bounds,
                action_bounds=action_bounds,
                resolution=resolution,
            ))
            optimal_qf_eval = create_optimal_qf(
                target_pos,
                state_bounds=state_bounds,
                action_bounds=action_bounds,
                resolution=resolution,
                discount_factor=discount_factor,
            )
            optimal_heatmaps.append(make_heat_map(
                optimal_qf_eval,
                state_bounds=state_bounds,
                action_bounds=action_bounds,
                resolution=resolution,
            ))

        fig = create_figure(heatmaps, optimal_heatmaps, target_poses,
                            iter_number)
        fig.savefig(str(save_file))

if __name__ == '__main__':
    main()
