from collections import defaultdict
from typing import NamedTuple

import numpy as np


def rlkit_buffer_to_macaw_format(buffer, discount_factor, path_length):
    size = buffer._top
    end_indices = compute_end_indices(buffer._terminals, size, path_length)
    data = {
        'obs': buffer._observations[:size],
        'actions': buffer._actions[:size],
        'rewards': buffer._rewards[:size],
        'next_obs': buffer._next_obs[:size],
        'terminals': buffer._terminals[:size],
        'discount_factor': discount_factor,
        'end_indices': end_indices,
    }
    for k, v in buffer._env_infos.items():
        data[k] = v[:size]
    add_trajectory_data_to_buffer(buffer, data, discount_factor, path_length)
    return data


def rlkit_buffer_to_borel_format(buffer, discount_factor, path_length):
    obs, actions, rewards, next_obs, terminals = [], [], [], [], []
    env_infos = defaultdict(list)
    for traj in yield_trajectories(buffer, path_length):
        if len(traj) != path_length:
            continue
        obs.append(np.array([e.state for e in traj]))
        actions.append(np.array([e.action for e in traj]))
        next_obs.append(np.array([e.next_state for e in traj]))
        rewards.append(np.array([e.reward for e in traj]))
        terminals.append(np.array([e.done for e in traj]))
        for k in buffer._env_infos.keys():
            env_infos[k].append(np.array([e.env_info[k] for e in traj]))
    data = {
        'obs': np.array(obs).transpose(1, 0, 2),
        'actions': np.array(actions).transpose(1, 0, 2),
        'rewards': np.array(rewards).transpose(1, 0, 2),
        'next_obs': np.array(next_obs).transpose(1, 0, 2),
        'terminals': np.array(terminals).transpose(1, 0, 2),
        'discount_factor': discount_factor,
        'trajectory_len': path_length,
    }
    for k in buffer._env_infos.keys():
        data[k] = np.array(env_infos[k]).transpose(1, 0, 2)

    return data


class Experience(NamedTuple):
    state: np.ndarray
    action: np.ndarray
    next_state: np.ndarray
    reward: float
    done: bool
    env_info: dict


def yield_trajectories(buffer, path_length):
    end_indices = compute_end_indices(buffer._terminals, buffer._top, path_length)
    current_traj = []
    for i in range(buffer._size):
        experience = Experience(
            state=buffer._observations[i],
            action=buffer._actions[i],
            next_state=buffer._next_obs[i],
            reward=buffer._rewards[i],
            done=buffer._terminals[i],
            env_info={k: infos[i] for k, infos in buffer._env_infos.items()}
        )
        current_traj.append(experience)
        if i in end_indices:
            yield current_traj
            current_traj = []


def add_trajectory_data_to_buffer(buffer, data, discount_factor, path_length):
    write_loc = 0
    all_terminal_obs = np.zeros_like(data['obs'])
    all_terminal_discounts = np.zeros_like(data['terminals'], dtype=np.float64)
    all_mc_rewards = np.zeros_like(data['rewards'])
    for trajectory in yield_trajectories(buffer, path_length):
        mc_reward = 0
        terminal_obs = None
        terminal_factor = 1
        for idx, experience in enumerate(trajectory[::-1]):
            if terminal_obs is None:
                terminal_obs = experience.next_state

            all_terminal_obs[write_loc] = terminal_obs
            terminal_factor *= discount_factor
            all_terminal_discounts[write_loc] = terminal_factor
            mc_reward = experience.reward + discount_factor * mc_reward
            all_mc_rewards[write_loc] = mc_reward
            write_loc += 1

    data['terminal_obs'] = all_terminal_obs
    data['terminal_discounts'] = all_terminal_discounts
    data['mc_rewards'] = all_mc_rewards


def compute_end_indices(terminals, size, path_length):
    """Return a list of end indices. A end index is an index where a new
    episode ends."""
    traj_start_i = 0
    current_i = 0
    end_indices = []
    while current_i < size:
        if (
            current_i - traj_start_i + 1 == path_length
            or terminals[current_i]
        ):
            end_indices.append(current_i)
            traj_start_i = current_i + 1
        current_i += 1
    return end_indices