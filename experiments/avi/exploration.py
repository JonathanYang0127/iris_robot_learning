import rlkit.torch.pytorch_util as ptu
from rlkit.misc.roboverse_utils import (add_reward_filtered_data_to_buffers_multitask, dump_video_basic,
    get_buffer_size)
from rlkit.exploration_strategies import *
from rlkit.samplers.data_collector import EmbeddingExplorationObsDictPathCollector

from roboverse.bullet.serializable import Serializable
import roboverse
import torch
import numpy as np
import argparse
import os
import gym


class EmbeddingWrapper(gym.Env, Serializable):

    def __init__(self, env, embeddings):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.embeddings = embeddings

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        obs.update({'task_embedding': self.embeddings[self.env.task_idx]})
        return obs, rew, done, info

    def reset(self):
        obs = self.env.reset()
        obs.update({'task_embedding': self.embeddings[self.env.task_idx]})
        return obs

    def reset_task(self, task_idx):
        self.env.reset_task(task_idx)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint-path", type=str, required=True)
    parser.add_argument("--env", type=str)
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--tsteps", type=int, default=20)
    parser.add_argument("--num-trajectories", type=int, default=50000)
    parser.add_argument("--action-noise", type=float, default=0.0)
    parser.add_argument("--video-save-frequency", type=int, default=20)
    parser.add_argument("--buffer", type=str, required=True)
    parser.add_argument('-e', '--exploration-strategy', type=str,
        choices=('gaussian', 'gaussian_filtered', 'cem', 'fast'))
    parser.add_argument('-u', '--update_frequency', type=int, default=20)
    parser.add_argument("--use-robot-state", default=False, action='store_true')
    args = parser.parse_args()

    expl_env = roboverse.make(args.env, transpose_image=True)
    ptu.set_gpu_mode(True)

    observation_keys = ['image', 'task_embedding']
    if args.use_robot_state:
        observation_keys.append('state')
        state_observation_dim = expl_env.observation_space.spaces['state'].low.size
    else:
        state_observation_dim = 0

    _, ext = os.path.splitext(args.checkpoint_path)
    with open(args.checkpoint_path, 'rb') as handle:
        if ext == ".pt":
            params = torch.load(handle)
            eval_policy = params['evaluation/policy']
            q_function = params['trainer/qf1']
        elif ext == ".pkl":
            eval_policy = pickle.load(handle)
    eval_policy.eval()

    with open(args.buffer, 'rb') as fl:
        data = np.load(fl, allow_pickle=True)

    embeddings = []
    for i in range(len(data)):
        for j in range(len(data[i]['observations'])):
            embeddings.append(data[i]['observations'][j]['task_embedding'])
    embeddings = np.array(embeddings)

    if args.exploration_strategy == 'gaussian':
        exploration_strategy = GaussianExplorationStrategy(embeddings, policy=eval_policy,
            q_function=q_function, n_components=10)
    elif args.exploration_strategy == 'gaussian_filtered':
        exploration_strategy = GaussianExplorationStrategy(embeddings, policy=eval_policy,
            q_function=q_function, n_components=10)
    elif args.exploration_strategy == 'cem':
        exploration_strategy = CEMExplorationStrategy(embeddings, update_frequency=args.update_frequency,
            n_components=10)
    elif args.exploration_strategy == 'fast':
        exploration_strategy = FastExplorationStrategy(embeddings, update_frequency=args.update_frequency,
            n_components=10)
    else:
        raise NotImplementedError

    expl_path_collector = EmbeddingExplorationObsDictPathCollector(
        exploration_strategy,
        expl_env,
        eval_policy,
        observation_keys=observation_keys,
    )

    paths = []
    for j in range(args.num_trajectories):
        new_paths = expl_path_collector.collect_new_paths(args.tsteps, args.tsteps,
            True, multi_task=True, task_index=args.task)

        paths.append(new_paths)
        if args.video_save_frequency != 0 and j % args.video_save_frequency == 0:
            dump_video_basic(f'videos/{j}', new_paths)
