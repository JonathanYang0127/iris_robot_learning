import abc
from collections import OrderedDict

import numpy as np

from railrl.misc.data_processing import create_stats_ordered_dict
from rllab.core.serializable import Serializable
from rllab.envs.proxy_env import ProxyEnv
from rllab.misc import logger
from rllab.spaces import Box


class MultitaskEnv(object, metaclass=abc.ABCMeta):
    """
    An environment with a task that can be specified with a goal state.
    """

    def __init__(self):
        self.multitask_goal = np.zeros(self.goal_dim)
        self.goal_dim_weights = np.ones(self.goal_dim)

    @property
    @abc.abstractmethod
    def goal_dim(self) -> int:
        """
        :return: int, dimension of goal state
        """
        pass

    @abc.abstractmethod
    def sample_goal_states(self, batch_size):
        pass

    @abc.abstractmethod
    def convert_obs_to_goal_states(self, obs):
        """
        Convert a raw environment observation into a goal state (if possible).
        """
        pass

    @abc.abstractmethod
    def sample_dimensions_irrelevant_to_oc(self, goal, obs, batch_size):
        """
        Create the OC goal state a bunch of time, but replace irrelevant goal
        dimensions with sampled values.

        :param goal: np.ndarray, shape GOAL_DIM
        :param batch_size:
        :return: ndarray, shape `batch_size` x GOAL_DIM
        """
        pass


    """
    Functions you probably don't need to override.
    """

    def sample_goal_state_for_rollout(self):
        """
        These goal states are fed to a policy when the policy wants to actually
        do rollouts.
        :return:
        """
        goal_state = self.sample_goal_states(1)[0]
        return self.modify_goal_state_for_rollout(goal_state)

    def convert_ob_to_goal_state(self, obs):
        """
        Convert a raw environment observation into a goal state (if possible).

        This observation should NOT include the goal state.
        """
        if isinstance(obs, np.ndarray):
            return self.convert_obs_to_goal_states(
                np.expand_dims(obs, 0)
            )[0]
        else:
            return self.convert_obs_to_goal_states_pytorch(
                obs.unsqueeze(0)
            )[0]

    """
    Check out these default functions below! You may want to override them.
    """
    def set_goal(self, goal):
        self.multitask_goal = goal

    def compute_rewards(self, obs, action, next_obs, goal_states):
        return - np.linalg.norm(
            self.convert_obs_to_goal_states(next_obs) - goal_states,
            axis=1,
            keepdims=True,
            ord=1,
        )

    def convert_obs_to_goal_states_pytorch(self, obs):
        """
        PyTorch version of `convert_obs_to_goal_state`.
        """
        return self.convert_obs_to_goal_states(obs)

    def modify_goal_state_for_rollout(self, goal_state):
        """
        Modify a goal state so that it's appropriate for doing a rollout.

        Common use case: zero out the goal velocities.
        :param goal_state:
        :return:
        """
        return goal_state

    def log_diagnostics(self, paths):
        statistics = OrderedDict()

        observations = np.vstack([path['observations'] for path in paths])
        goal_states = np.vstack([path['goal_states'] for path in paths])
        actions = np.vstack([path['actions'] for path in paths])
        final_differences = []
        for path in paths:
            reached = self.convert_ob_to_goal_state(path['observations'][-1])
            goal = path['goal_states'][-1]
            final_differences.append(reached - goal)
        for order in [1, 2]:
            final_distances = np.linalg.norm(
                np.array(final_differences),
                axis=1,
                ord=order,
            )
            goal_distances = np.linalg.norm(
                self.convert_obs_to_goal_states(observations) - goal_states,
                axis=1,
                ord=order,
            )
            statistics.update(create_stats_ordered_dict(
                'Multitask L{} distance to goal'.format(order),
                goal_distances,
                always_show_all_stats=True,
            ))
            statistics.update(create_stats_ordered_dict(
                'Multitask Final L{} distance to goal'.format(order),
                final_distances,
                always_show_all_stats=True,
            ))
        rewards = self.compute_rewards(
            observations[:-1, ...],
            actions[:-1, ...],
            observations[1:, ...],
            goal_states[:-1, ...],
        )
        statistics.update(create_stats_ordered_dict(
            'Multitask Env Rewards', rewards,
        ))
        for key, value in statistics.items():
            logger.record_tabular(key, value)

    """
    Optional functions to implement, since most of my code doesn't use these
    any more.
    """
    def sample_irrelevant_goal_dimensions(self, goal, batch_size):
        """
        Copy the goal a bunch of time, but replace irrelevant goal dimensions
        with sampled values.

        For example, if you care about the position but not about the velocity,
        copy the velocity `batch_size` number of times, and then sample a bunch
        of velocity values.

        :param goal: np.ndarray, shape GOAL_DIM
        :param batch_size:
        :return: ndarray, shape SAMPLE_SIZE x GOAL_DIM
        """
        pass

    def sample_actions(self, batch_size):
        pass

    def sample_states(self, batch_size):
        pass



class MultitaskToFlatEnv(ProxyEnv, Serializable):
    def __init__(
            self,
            env: MultitaskEnv,
    ):
        # self._wrapped_env needs to be called first because
        # Serializable.quick_init calls getattr, on this class. And the
        # implementation of getattr (see below) calls self._wrapped_env.
        # Without setting this first, the call to self._wrapped_env would call
        # getattr again (since it's not set yet) and therefore loop forever.
        self._wrapped_env = env
        # Or else serialization gets delegated to the wrapped_env. Serialize
        # this env separately from the wrapped_env.
        self._serializable_initialized = False
        Serializable.quick_init(self, locals())
        ProxyEnv.__init__(self, env)

    @property
    def observation_space(self):
        wrapped_low = super().observation_space.low
        low = np.hstack((
            wrapped_low,
            min(wrapped_low) * np.ones(self._wrapped_env.goal_dim)
        ))
        wrapped_high = super().observation_space.low
        high = np.hstack((
            wrapped_high,
            max(wrapped_high) * np.ones(self._wrapped_env.goal_dim)
        ))
        return Box(low, high)

    def step(self, action):
        ob, reward, done, info_dict = self._wrapped_env.step(action)
        ob = np.hstack((ob, self._wrapped_env.multitask_goal))
        return ob, reward, done, info_dict

    def reset(self):
        ob = super().reset()
        ob = np.hstack((ob, self._wrapped_env.multitask_goal))
        return ob

multitask_to_flat_env = MultitaskToFlatEnv
