"""
Plot histogram of the actions and observations in a replay buffer.
"""
import joblib

import numpy as np
import argparse
import pickle

from railrl.envs.multitask.reacher_env import (
    XyMultitaskSimpleStateReacherEnv,
    GoalStateSimpleStateReacherEnv,
)
import matplotlib.pyplot as plt


def main(dataset_path, load_joblib=True):
    if load_joblib:
        data = joblib.load(dataset_path)
        replay_buffer = data['replay_buffer']
        env = data['env']
    else:
        env = XyMultitaskSimpleStateReacherEnv()
        with open(dataset_path, 'rb') as handle:
            replay_buffer = pickle.load(handle)

    train_replay_buffer = replay_buffer.train_replay_buffer
    actions = train_replay_buffer._actions
    action_dim = actions.shape[-1]
    fig, axes = plt.subplots(action_dim)
    for i in range(action_dim):
        ax = axes[i]
        x = actions[:train_replay_buffer._size, i]
        ax.hist(x)
        ax.set_title("actions, dim #{}".format(i+1))
    plt.show()

    obs = train_replay_buffer._observations
    num_features = obs.shape[-1]
    fig, axes = plt.subplots(num_features)
    for i in range(num_features):
        ax = axes[i]
        x = obs[:train_replay_buffer._size, i]
        ax.hist(x)
        ax.set_title("observations, dim #{}".format(i+1))
    plt.show()

    batch_size = 100
    batch = train_replay_buffer.random_batch(batch_size)
    sampled_goal_states = env.sample_goal_states(batch_size)
    computed_rewards = env.compute_rewards(
        batch['observations'],
        batch['actions'],
        batch['next_observations'],
        sampled_goal_states
    )
    fig, ax = plt.subplots(1)
    ax.hist(computed_rewards)
    ax.set_title("computed rewards")
    plt.show()

    if isinstance(env, GoalStateSimpleStateReacherEnv):
        differences = batch['next_observations'] - sampled_goal_states
        num_features = differences.shape[-1]
        fig, axes = plt.subplots(num_features)
        for i in range(num_features):
            ax = axes[i]
            x = differences[:, i]
            ax.hist(x)
            ax.set_title("next_obs - goal state, dim #{}".format(i+1))
        plt.show()
    import ipdb; ipdb.set_trace()
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('replay_pkl_path', type=str,
                        help='path to the snapshot file')
    args = parser.parse_args()

    dataset_path = args.replay_pkl_path
    main(dataset_path)
