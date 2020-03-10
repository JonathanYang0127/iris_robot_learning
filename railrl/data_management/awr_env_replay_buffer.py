import torch
from gym.spaces import Discrete
import numpy as np
from railrl.data_management.simple_replay_buffer import SimpleReplayBuffer
from railrl.envs.env_utils import get_dim
from railrl.torch import pytorch_util as ptu
import torch.nn.functional as F


class AWREnvReplayBuffer(SimpleReplayBuffer):
    def __init__(
            self,
            max_replay_buffer_size,
            env,
            env_info_sizes=None,
            use_weights=False,
            policy=None,
            qf1=None,
            beta=0,
            weight_update_period=10000,
    ):
        """
        :param max_replay_buffer_size:
        :param env:
        """
        self.env = env
        self._ob_space = env.observation_space
        self._action_space = env.action_space

        if env_info_sizes is None:
            if hasattr(env, 'info_sizes'):
                env_info_sizes = env.info_sizes
            else:
                env_info_sizes = dict()
        self._use_weights = use_weights
        self.policy = policy
        self.qf1 = qf1
        self.beta = beta
        self.weight_update_period = weight_update_period
        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size,
            observation_dim=get_dim(self._ob_space),
            action_dim=get_dim(self._action_space),
            env_info_sizes=env_info_sizes
        )
        self.weights = torch.zeros((self._max_replay_buffer_size, 1), dtype=torch.float32)
        self.actual_weights = None
        self.counter = 0

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        if isinstance(self._action_space, Discrete):
            new_action = np.zeros(self._action_dim)
            new_action[action] = 1
        else:
            new_action = action
        return super().add_sample(
            observation=observation,
            action=new_action,
            reward=reward,
            next_observation=next_observation,
            terminal=terminal,
            **kwargs
        )

    def refresh_weights(self):
        if self.actual_weights is None or (self.counter % self.weight_update_period == 0 and self._use_weights):
            batch_size = 1024
            next_idx = min(batch_size, self._size)

            cur_idx = 0
            while cur_idx < self._size:
                idxs = np.arange(cur_idx, next_idx)
                obs = ptu.from_numpy(self._observations[idxs])
                actions = ptu.from_numpy(self._actions[idxs])

                new_obs_actions, policy_mean, policy_log_std, log_pi, entropy, policy_std, *_ = self.policy(
                    obs, reparameterize=True, return_log_prob=True,
                )
                q1_pred = self.qf1(obs, actions)
                v_pi = self.qf1(obs, new_obs_actions)

                advantage = q1_pred - v_pi
                self.weights[idxs] = (advantage/self.beta).detach()

                cur_idx = next_idx
                next_idx += batch_size
                next_idx = min(next_idx, self._size)

            self.actual_weights = ptu.get_numpy(F.softmax(self.weights[:self._size], dim=0))
            p_sum = np.sum(self.actual_weights)
            assert p_sum > 0, "Unnormalized p sum is {}".format(p_sum)
            self.actual_weights = (self.actual_weights/p_sum).flatten()
        self.counter += 1

    def sample_weighted_indices(self, batch_size):
        if self._use_weights:
            indices = np.random.choice(
                len(self.actual_weights),
                batch_size,
                p=self.actual_weights,
            )
        else:
            indices = self._sample_indices(batch_size)
        return indices

    def _sample_indices(self, batch_size):
        return np.random.randint(0, self._size, batch_size)

    def random_batch(self, batch_size):
        self.refresh_weights()
        indices = self.sample_weighted_indices(batch_size)
        if self._use_weights:
            weights = self.actual_weights[indices]
        else:
            weights = self._rewards[indices]
        batch = dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._next_obs[indices],
            weights=weights,
        )
        for key in self._env_info_keys:
            assert key not in batch.keys()
            batch[key] = self._env_infos[key][indices]
        return batch