from pathlib import Path
from typing import NamedTuple, List

from rlkit.misc.asset_loader import (
    load_local_or_remote_file,
    local_path_from_s3_or_local_path,
)
import numpy as np


def load_buffer_onto_algo(
        pretrain_buffer_path,
):
    data = load_local_or_remote_file(
        pretrain_buffer_path,
        file_type='joblib',
    )
    saved_replay_buffer = data['replay_buffer']
    save_dir = Path(
        local_path_from_s3_or_local_path(pretrain_buffer_path)
    ).parent
    for k in saved_replay_buffer.task_buffers:
        buffer = saved_replay_buffer.task_buffers[k]
        data = convert_buffer(buffer)
        save_path = save_dir / 'converted_task_{}.npy'.format(k)
        print('saving to', save_path)
        np.save(save_path, data)


def convert_buffer(buffer):
    size = buffer._top
    data = {
        'obs': buffer._observations[:size],
        'actions': buffer._actions[:size],
        'rewards': buffer._rewards[:size],
        'next_obs': buffer._next_obs[:size],
        'terminals': buffer._terminals[:size],
        'discount_factor': discount_factor,
    }

    add_trajectory_data_to_buffer(buffer, data)
    return data


class Experience(NamedTuple):
    state: np.ndarray
    action: np.ndarray
    next_state: np.ndarray
    reward: float
    done: bool


def yield_trajectories(buffer):
    end_indices = compute_end_indices(buffer)
    current_traj = []
    for i in range(buffer._size):
        experience = Experience(
            state=buffer._observations[i],
            action=buffer._actions[i],
            next_state=buffer._next_obs[i],
            reward=buffer._rewards[i],
            done=buffer._terminals[i],
        )
        current_traj.append(experience)
        if i in end_indices:
            yield current_traj
            current_traj = []


def add_trajectory_data_to_buffer(buffer, data):
    write_loc = 0
    all_terminal_obs = np.zeros_like(data['obs'])
    all_terminal_discounts = np.zeros_like(data['terminals'])
    all_mc_rewards = np.zeros_like(data['rewards'])
    for trajectory in yield_trajectories(buffer):
        mc_reward = 0
        terminal_obs = None
        terminal_factor = 1
        for idx, experience in enumerate(trajectory[::-1]):
            if terminal_obs is None:
                terminal_obs = experience.next_state

            all_terminal_obs[write_loc] = terminal_obs
            terminal_factor *= discount_factor
            all_terminal_discounts[write_loc] = terminal_factor
            all_mc_rewards[write_loc] = mc_reward

    data['terminal_obs'] = all_terminal_obs
    data['terminal_discounts'] = all_terminal_discounts
    data['mc_rewards'] = all_mc_rewards
    return data


def compute_end_indices(buffer):
    """Return a list of end indices. A end index is an index where a new
    episode ends."""
    size = buffer._top
    traj_start_i = 0
    current_i = 0
    end_indices = []
    while current_i < size:
        if (
            current_i - traj_start_i + 1 == path_length
            or buffer._terminals[current_i]
        ):
            end_indices.append(current_i)
            traj_start_i = current_i + 1
        current_i += 1
    return end_indices


if __name__ == '__main__':
    path_length = 200
    discount_factor = 0.99
    pretrain_buffer_path = "21-02-22-ant-awac--exp7-ant-dir-4-eval-4-train-sac-to-get-buffer-longer/21-02-22-ant-awac--exp7-ant-dir-4-eval-4-train-sac-to-get-buffer-longer_2021_02_23_06_09_23_id000--s270987/extra_snapshot_itr400.cpkl"
    load_buffer_onto_algo(
        pretrain_buffer_path
    )
