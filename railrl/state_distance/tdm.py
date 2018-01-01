"""
New implementation of state distance q learning.
"""
import abc

import numpy as np

from railrl.data_management.path_builder import PathBuilder
from railrl.envs.remote import RemoteRolloutEnv
from railrl.misc.np_util import truncated_geometric
from railrl.misc.ml_util import ConstantSchedule
from railrl.policies.base import SerializablePolicy
from railrl.policies.state_distance import UniversalPolicy
from railrl.state_distance.exploration import MakeUniversal
from railrl.state_distance.rollout_util import MultigoalSimplePathSampler, \
    multitask_rollout
from railrl.state_distance.tdm_networks import TdmNormalizer
from railrl.state_distance.util import merge_into_flat_obs
from railrl.torch.algos.torch_rl_algorithm import TorchRLAlgorithm
from railrl.torch.algos.util import np_to_pytorch_batch
from railrl.torch.data_management.normalizer import TorchFixedNormalizer


class TemporalDifferenceModel(TorchRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
            self,
            max_tau=10,
            epoch_max_tau_schedule=None,
            sample_train_goals_from='replay_buffer',
            sample_rollout_goals_from='environment',
            vectorized=True,
            cycle_taus_for_rollout=False,
            dense_rewards=False,
            finite_horizon=True,
            tau_sample_strategy='uniform',
            reward_type='distance',
            goal_reached_epsilon=1e-3,
            terminate_when_goal_reached=False,
            truncated_geom_factor=2.,
            norm_order=1,
            goal_weights=None,
            tdm_normalizer: TdmNormalizer=None,
            num_paths_for_normalization=0,
            normalize_distance=False,
    ):
        """

        :param max_tau: Maximum tau (planning horizon) to train with.
        :param epoch_max_tau_schedule: A schedule for the maximum planning
        horizon tau.
        :param sample_train_goals_from: Sampling strategy for goals used in
        training. Can be one of the following strings:
            - environment: Sample from the environment
            - replay_buffer: Sample from anywhere in the replay_buffer
            - her: Sample from a HER-based replay_buffer
            - no_resampling: Use the goals used in the rollout
        :param sample_rollout_goals_from: Sampling strategy for goals used
        during rollout. Can be one of the following strings:
            - environment: Sample from the environment
            - replay_buffer: Sample from the replay_buffer
            - fixed: Do no resample the goal. Just use the one in the
            environment.
        :param vectorized: Train the QF in vectorized form?
        :param cycle_taus_for_rollout: Decrement the tau passed into the
        policy during rollout?
        :param dense_rewards: If True, always give rewards. Otherwise,
        only give rewards when the episode terminates.
        :param finite_horizon: If True, use a finite horizon formulation:
        give the time as input to the Q-function and terminate.
        :param tau_sample_strategy: Sampling strategy for taus used
        during training. Can be one of the following strings:
            - no_resampling: Do not resample the tau. Use the one from rollout.
            - uniform: Sample uniformly from [0, max_tau]
            - truncated_geometric: Sample from a truncated geometric
            distribution, truncated at max_tau.
            - all_valid: Always use all 0 to max_tau values
        :param reward_type: One of the following:
            - 'distance': Reward is -|s_t - s_g|
            - 'indicator': Reward is -1{||s_t - s_g||_2 > epsilon}
            - 'env': Use the environment reward
        :param goal_reached_epsilon: Epsilon used to determine if the goal
        has been reached. Used by `indicator` version of `reward_type` and when
        `terminate_whe_goal_reached` is True.
        :param terminate_when_goal_reached: Do you terminate when you have
        reached the goal?
        :param norm_order: If vectorized=False, do you use L1, L2,
        etc. for distance?
        :param goal_weights: None or the weights for the different goal
        dimensions. These weights are used to compute the distances to the goal.
        """
        assert sample_train_goals_from in ['environment', 'replay_buffer',
                                           'her', 'no_resampling']
        assert sample_rollout_goals_from in ['environment', 'replay_buffer',
                                             'fixed']
        assert tau_sample_strategy in [
            'no_resampling',
            'uniform',
            'truncated_geometric',
            'all_valid',
        ]
        assert reward_type in ['distance', 'indicator', 'env']
        if epoch_max_tau_schedule is None:
            epoch_max_tau_schedule = ConstantSchedule(max_tau)

        if not finite_horizon:
            max_tau = 0
            epoch_max_tau_schedule = ConstantSchedule(max_tau)
            cycle_taus_for_rollout = False

        self.max_tau = max_tau
        self.epoch_max_tau_schedule = epoch_max_tau_schedule
        self.sample_train_goals_from = sample_train_goals_from
        self.sample_rollout_goals_from = sample_rollout_goals_from
        self.vectorized = vectorized
        self.cycle_taus_for_rollout = cycle_taus_for_rollout
        self.dense_rewards = dense_rewards
        self.finite_horizon = finite_horizon
        self.tau_sample_strategy = tau_sample_strategy
        self.reward_type = reward_type
        self.goal_reached_epsilon = goal_reached_epsilon
        self.terminate_when_goal_reached = terminate_when_goal_reached
        self.norm_order = norm_order
        self._current_path_goal = None
        self._rollout_tau = self.max_tau
        self.truncated_geom_factor = float(truncated_geom_factor)
        self.goal_weights = goal_weights
        if self.goal_weights is not None:
            # In case they were passed in as (e.g.) tuples or list
            self.goal_weights = np.array(self.goal_weights)
            assert self.goal_weights.size == self.env.goal_dim
        self.tdm_normalizer = tdm_normalizer
        self.num_paths_for_normalization = num_paths_for_normalization
        self.normalize_distance = normalize_distance

        self.policy = MakeUniversal(self.policy)
        self.eval_policy = MakeUniversal(self.eval_policy)
        self.exploration_policy = MakeUniversal(self.exploration_policy)
        self.eval_sampler = MultigoalSimplePathSampler(
            env=self.env,
            policy=self.eval_policy,
            max_samples=self.num_steps_per_eval,
            max_path_length=self.max_path_length,
            tau_sampling_function=self._sample_max_tau_for_rollout,
            goal_sampling_function=self._sample_goal_for_rollout,
            cycle_taus_for_rollout=self.cycle_taus_for_rollout,
        )
        if self.collection_mode == 'online-parallel':
            # TODO(murtaza): What happens to the eval env?
            # see `eval_sampler` definition above.
            self.training_env = RemoteRolloutEnv(
                env=self.env,
                policy=self.eval_policy,
                exploration_policy=self.exploration_policy,
                max_path_length=self.max_path_length,
                normalize_env=self.normalize_env,
                rollout_function=self.rollout,
            )

    def _start_epoch(self, epoch):
        self.max_tau = self.epoch_max_tau_schedule.get_value(epoch)
        super()._start_epoch(epoch)

    def get_batch(self, training=True):
        if self.replay_buffer_is_split:
            replay_buffer = self.replay_buffer.get_replay_buffer(training)
        else:
            replay_buffer = self.replay_buffer
        batch = replay_buffer.random_batch(self.batch_size)

        """
        Update the goal states/rewards
        """
        num_steps_left = self._sample_taus_for_training(batch)
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        goals = self._sample_goals_for_training(batch)
        rewards = self._compute_scaled_rewards_np(batch, obs, actions, next_obs, goals)
        terminals = batch['terminals']

        if self.tau_sample_strategy == 'all_valid':
            obs = np.repeat(obs, self.max_tau + 1, 0)
            actions = np.repeat(actions, self.max_tau + 1, 0)
            next_obs = np.repeat(next_obs, self.max_tau + 1, 0)
            goals = np.repeat(goals, self.max_tau + 1, 0)
            rewards = np.repeat(rewards, self.max_tau + 1, 0)
            terminals = np.repeat(terminals, self.max_tau + 1, 0)

        if self.finite_horizon:
            terminals = 1 - (1 - terminals) * (num_steps_left != 0)
        if self.terminate_when_goal_reached:
            diff = self.env.convert_obs_to_goals(next_obs) - goals
            goal_not_reached = (
                np.linalg.norm(diff, axis=1, keepdims=True)
                > self.goal_reached_epsilon
            )
            terminals = 1 - (1 - terminals) * goal_not_reached

        if not self.dense_rewards:
            rewards = rewards * terminals

        """
        Update the batch
        """
        batch['rewards'] = rewards
        batch['terminals'] = terminals
        batch['actions'] = actions
        batch['observations'] = merge_into_flat_obs(
            obs=obs,
            goals=goals,
            num_steps_left=num_steps_left,
        )
        if self.finite_horizon:
            batch['next_observations'] = merge_into_flat_obs(
                obs=next_obs,
                goals=goals,
                num_steps_left=num_steps_left-1,
            )
        else:
            batch['next_observations'] = merge_into_flat_obs(
                obs=next_obs,
                goals=goals,
                num_steps_left=num_steps_left,
            )

        return np_to_pytorch_batch(batch)

    def _compute_scaled_rewards_np(self, batch, obs, actions, next_obs, goals):
        """
        Rewards should be already multiplied by the reward scale and/or other
        factors. In other words, the rewards returned here should be
        immediately ready for any down-stream learner to consume.
        """
        if self.reward_type == 'indicator':
            diff = self.env.convert_obs_to_goals(next_obs) - goals
            if self.vectorized:
                return -self.reward_scale * (diff > self.goal_reached_epsilon)
            else:
                return -self.reward_scale * (
                    np.linalg.norm(diff, axis=1, keepdims=True)
                    > self.goal_reached_epsilon
                )
        elif self.reward_type == 'distance':
            neg_distances = self._compute_raw_neg_distances(next_obs, goals)
            if self.goal_weights is not None:
                neg_distances = neg_distances * self.goal_weights
            return neg_distances * self.reward_scale
        elif self.reward_type == 'env':
            return batch['rewards']
        else:
            raise TypeError("Invalid reward type: {}".format(self.reward_type))

    def _compute_raw_neg_distances(self, next_obs, goals):
        diff = self.env.convert_obs_to_goals(next_obs) - goals
        if self.vectorized:
            raw_neg_distances = -np.abs(diff)
        else:
            raw_neg_distances = -np.linalg.norm(
                diff,
                ord=self.norm_order,
                axis=1,
                keepdims=True,
            )
        return raw_neg_distances

    @property
    def train_buffer(self):
        if self.replay_buffer_is_split:
            return self.replay_buffer.get_replay_buffer(trainig=True)
        else:
            return self.replay_buffer

    def _sample_taus_for_training(self, batch):
        if self.finite_horizon:
            if self.tau_sample_strategy == 'uniform':
                num_steps_left = np.random.randint(
                    0, self.max_tau + 1, (self.batch_size, 1)
                )
            elif self.tau_sample_strategy == 'truncated_geometric':
                num_steps_left = truncated_geometric(
                    p=self.truncated_geom_factor/self.max_tau,
                    truncate_threshold=self.max_tau,
                    size=(self.batch_size, 1),
                    new_value=0
                )
            elif self.tau_sample_strategy == 'no_resampling':
                num_steps_left = batch['num_steps_left']
            elif self.tau_sample_strategy == 'all_valid':
                num_steps_left = np.tile(
                    np.arange(0, self.max_tau+1),
                    self.batch_size
                )
                num_steps_left = np.expand_dims(num_steps_left, 1)
            else:
                raise TypeError("Invalid tau_sample_strategy: {}".format(
                    self.tau_sample_strategy
                ))
        else:
            num_steps_left = np.zeros((self.batch_size, 1))
        return num_steps_left

    def _sample_goals_for_training(self, batch):
        if self.sample_train_goals_from == 'her':
            return batch['resampled_goals']
        elif self.sample_train_goals_from == 'no_resampling':
            return batch['goals_used_for_rollout']
        elif self.sample_train_goals_from == 'environment':
            return self.env.sample_goals(self.batch_size)
        elif self.sample_train_goals_from == 'replay_buffer':
            batch = self.train_buffer.random_batch(self.batch_size)
            obs = batch['observations']
            return self.env.convert_obs_to_goals(obs)
        else:
            raise Exception("Invalid `sample_train_goals_from`: {}".format(
                self.sample_train_goals_from
            ))

    def _sample_goal_for_rollout(self):
        if self.sample_rollout_goals_from == 'environment':
            return self.env.sample_goal_for_rollout()
        elif self.sample_rollout_goals_from == 'replay_buffer':
            batch = self.train_buffer.random_batch(1)
            obs = batch['observations']
            goal = self.env.convert_obs_to_goals(obs)[0]
            return self.env.modify_goal_for_rollout(goal)
        elif self.sample_rollout_goals_from == 'fixed':
            return self.env.multitask_goal
        else:
            raise Exception("Invalid `sample_goals_from`: {}".format(
                self.sample_rollout_goals_from
            ))

    def _sample_max_tau_for_rollout(self):
        if self.finite_horizon:
            return self.max_tau
        else:
            return 0

    def offline_evaluate(self, epoch):
        raise NotImplementedError()

    def _start_new_rollout(self):
        self.exploration_policy.reset()
        self._current_path_goal = self._sample_goal_for_rollout()
        self.training_env.set_goal(self._current_path_goal)
        self.exploration_policy.set_goal(self._current_path_goal)
        self._rollout_tau = self.max_tau
        self.exploration_policy.set_tau(self._rollout_tau)
        return self.training_env.reset()

    def _handle_step(
            self,
            observation,
            action,
            reward,
            next_observation,
            terminal,
            agent_info,
            env_info,
    ):
        self._current_path_builder.add_all(
            observations=observation,
            actions=action,
            rewards=reward,
            next_observations=next_observation,
            terminals=terminal,
            agent_infos=agent_info,
            env_infos=env_info,
            num_steps_left=np.array([self._rollout_tau]),
            goals=self._current_path_goal,
        )
        if self.cycle_taus_for_rollout:
            self._rollout_tau -= 1
            if self._rollout_tau < 0:
                self._rollout_tau = self.max_tau
            self.exploration_policy.set_tau(self._rollout_tau)

    def _handle_rollout_ending(self):
        self._n_rollouts_total += 1
        if len(self._current_path_builder) > 0:
            path = self._current_path_builder.get_all_stacked()
            self.replay_buffer.add_path(path)
            self._exploration_paths.append(path)
            self._current_path_builder = PathBuilder()

    def _handle_path(self, path):
        self._n_rollouts_total += 1
        self.replay_buffer.add_path(path)

    def pretrain(self):
        if self.num_paths_for_normalization == 0:
            return

        paths = []
        random_policy = RandomUniveralPolicy(self.env.action_space)
        while len(paths) < self.num_paths_for_normalization:
            goal = self._sample_goal_for_rollout()
            path = multitask_rollout(
                self.training_env,
                random_policy,
                goal=goal,
                tau=0,
                max_path_length=self.max_path_length,
            )
            paths.append(path)

        goals = np.vstack([
            self._sample_goal_for_rollout()
            for _ in range(
                self.num_paths_for_normalization * self.max_path_length
            )
        ])
        obs = np.vstack([path["observations"] for path in paths])
        next_obs = np.vstack([path["next_observations"] for path in paths])
        actions = np.vstack([path["actions"] for path in paths])
        neg_distances = self._compute_raw_neg_distances(next_obs, goals)

        ob_mean = np.mean(obs, axis=0)
        ob_std = np.std(obs, axis=0)
        ac_mean = np.mean(actions, axis=0)
        ac_std = np.std(actions, axis=0)
        goal_mean = np.mean(goals, axis=0)
        goal_std = np.std(goals, axis=0)
        distance_mean = np.mean(neg_distances, axis=0)
        distance_std = np.std(neg_distances, axis=0)

        if self.tdm_normalizer is not None:
            self.tdm_normalizer.obs_normalizer.set_mean(ob_mean)
            self.tdm_normalizer.obs_normalizer.set_std(ob_std)
            self.tdm_normalizer.action_normalizer.set_mean(ac_mean)
            self.tdm_normalizer.action_normalizer.set_std(ac_std)
            self.tdm_normalizer.goal_normalizer.set_mean(goal_mean)
            self.tdm_normalizer.goal_normalizer.set_std(goal_std)
            if self.normalize_distance:
                self.tdm_normalizer.distance_normalizer.set_mean(distance_mean)
                self.tdm_normalizer.distance_normalizer.set_std(distance_std)


class RandomUniveralPolicy(UniversalPolicy, SerializablePolicy):
    """
    Policy that always outputs zero.
    """

    def __init__(self, action_space):
        super().__init__()
        self.action_space = action_space

    def get_action(self, obs):
        return self.action_space.sample(), {}
