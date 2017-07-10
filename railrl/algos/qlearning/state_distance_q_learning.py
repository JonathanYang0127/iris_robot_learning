from collections import OrderedDict

import pickle
import random
import os
import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

from railrl.data_management.env_replay_buffer import EnvReplayBuffer
from railrl.data_management.split_buffer import SplitReplayBuffer
from railrl.misc.data_processing import create_stats_ordered_dict
from railrl.misc.rllab_util import get_average_returns, split_paths
from railrl.pythonplusplus import identity
from railrl.torch.core import PyTorchModule
from railrl.torch.ddpg import DDPG, np_to_pytorch_batch
from railrl.torch.online_algorithm import OnlineAlgorithm
import railrl.torch.pytorch_util as ptu
from rllab.misc import logger, special


class StateDistanceQLearning(DDPG):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

    def get_batch(self, training=True):
        pool = self.pool.get_replay_buffer(training)
        batch_size = min(
            pool.num_steps_can_sample(),
            self.batch_size
        )
        batch = pool.random_batch(batch_size)
        goal_states = self.env.sample_goal_states(batch_size)
        new_rewards = self.env.compute_rewards(
            batch['observations'],
            batch['actions'],
            batch['next_observations'],
            goal_states,
        )
        batch['observations'] = batch['observations'], goal_states
        batch['next_observations'] = (
            batch['next_observations'], goal_states
        )
        batch['rewards'] = new_rewards
        torch_batch = np_to_pytorch_batch(batch)
        torch_batch['observations'] = torch.cat(
            torch_batch['observations'], dim=1
        )
        torch_batch['next_observations'] = torch.cat(
            torch_batch['next_observations'], dim=1
        )
        return torch_batch

    def reset_env(self):
        self.exploration_strategy.reset()
        self.exploration_policy.reset()
        self.policy.reset()
        return self.training_env.reset()

    def _paths_to_np_batch(self, paths):
        batch = super()._paths_to_np_batch(paths)
        batch_size = len(batch['observations'])
        goal_states = self.env.sample_goal_states(batch_size)
        new_rewards = self.env.compute_rewards(
            batch['observations'],
            batch['actions'],
            batch['next_observations'],
            goal_states,
        )
        batch['observations'] = np.hstack((batch['observations'], goal_states))
        batch['next_observations'] = np.hstack((
            batch['next_observations'], goal_states
        ))
        batch['rewards'] = new_rewards
        return batch


class MultitaskPathSampler(object):
    def __init__(
            self,
            env,
            exploration_policy,
            exploration_strategy,
            pool,
            min_num_steps_to_collect=1000,
            max_path_length=None,
    ):
        self.env = env
        self.exploration_policy = exploration_policy
        self.exploration_strategy = exploration_strategy
        self.min_num_steps_to_collect = min_num_steps_to_collect
        self.pool = pool
        if max_path_length is None:
            max_path_length = np.inf
        self.max_path_length = max_path_length

    def collect_data(self):
        obs = self.env.reset()
        n_steps_total = 0
        path_length = 0
        while n_steps_total < self.min_num_steps_to_collect:
            action, agent_info = (
                self.exploration_strategy.get_action(
                    n_steps_total,
                    obs,
                    self.exploration_policy,
                )
            )

            next_ob, raw_reward, terminal, env_info = (
                self.env.step(action)
            )
            n_steps_total += 1
            path_length += 1
            reward = raw_reward

            self.pool.add_sample(
                obs,
                action,
                reward,
                terminal,
                agent_info=agent_info,
                env_info=env_info,
            )
            if terminal or path_length >= self.max_path_length:
                self.pool.terminate_episode(
                    next_ob,
                    terminal,
                    agent_info=agent_info,
                    env_info=env_info,
                )
                obs = self.reset_env()
                path_length = 0
            else:
                obs = next_ob

    def save_pool(self, filename):
        # train_file = os.path.join(dir_name, 'train.pkl')
        # validation_file = os.path.join(dir_name, 'validation.pkl')
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as handle:
            pickle.dump(self.pool, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def reset_env(self):
        self.exploration_strategy.reset()
        self.exploration_policy.reset()
        return self.env.reset()


class StateDistanceQLearningSimple(StateDistanceQLearning):
    def __init__(
            self,
            *args,
            pool=None,
            num_batches=100,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.num_batches = num_batches
        self.pool = pool

    def train(self):
        for n_steps_total in range(self.num_batches):
            self.training_mode(True)
            self._do_training(n_steps_total=n_steps_total)
            if n_steps_total % 100 == 0:
                self.training_mode(False)
                self.evaluate(n_steps_total, None)

    def evaluate(self, epoch, _):
        """
        Perform evaluation for this algorithm.

        :param epoch: The epoch number.
        :param exploration_paths: List of dicts, each representing a path.
        """
        statistics = OrderedDict()
        train_batch = self.get_batch(training=True)
        statistics.update(self._statistics_from_batch(train_batch, "Train"))
        validation_batch = self.get_batch(training=False)
        statistics.update(
            self._statistics_from_batch(validation_batch, "Validation")
        )

        statistics['QF Loss Validation - Train Gap'] = (
            statistics['Validation QF Loss Mean']
            - statistics['Train QF Loss Mean']
        )
        statistics['Policy Loss Validation - Train Gap'] = (
            statistics['Validation Policy Loss Mean']
            - statistics['Train Policy Loss Mean']
        )
        for key, value in statistics.items():
            logger.record_tabular(key, value)


class ArgmaxPolicy(PyTorchModule):
    def __init__(
            self,
            qf,
            num_actions,
            prob_random_action=0.1,
    ):
        self.save_init_params(locals())
        super().__init__()
        self.qf = qf
        self.num_actions = num_actions
        self.prob_random_action = prob_random_action

    def get_action(self, obs):
        if random.random() <= self.prob_random_action:
            return random.randint(0, self.num_actions)
        obs = np.expand_dims(obs, axis=0)
        obs = Variable(ptu.from_numpy(obs).float(), requires_grad=False)
        qvalues = self.qf(obs)
        action = torch.max(qvalues)[1]
        return action, {}


class DQN(PyTorchModule):
    def __init__(
            self,
            obs_dim,
            action_dim,
            fc1_size,
            fc2_size,
            init_w=3e-3,
            output_activation=identity,
            hidden_init=ptu.fanin_init,
    ):
        self.save_init_params(locals())
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.output_activation = output_activation
        self.hidden_init = hidden_init

        self.fc1 = nn.Linear(obs_dim, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.last_fc = nn.Linear(fc2_size, action_dim)

        self.hidden_init(self.obs_fc.weight)
        self.obs_fc.bias.data.fill_(0)
        self.hidden_init(self.embedded_fc.weight)
        self.embedded_fc.bias.data.fill_(0)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, obs, action):
        h = obs
        h = F.relu(self.obs_fc(h))
        h = torch.cat((h, action), dim=1)
        h = F.relu(self.embedded_fc(h))
        return self.output_activation(self.last_fc(h))
