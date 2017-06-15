from railrl.exploration_strategies.base import RawExplorationStrategy
from rllab.core.serializable import Serializable
from rllab.spaces.box import Box
import numpy as np
import numpy.random as nr


class GaussianStrategy(RawExplorationStrategy, Serializable):
    """
    This strategy adds Gaussian noise to the action taken by the deterministic policy.

    Based on the rllab implementation.
    """
    def __init__(self, action_space, max_sigma=1.0, min_sigma=0.1,
                 decay_period=1000000):
        assert isinstance(action_space, Box)
        assert len(action_space.shape) == 1
        Serializable.quick_init(self, locals())
        self._max_sigma = max_sigma
        self._min_sigma = min_sigma
        self._decay_period = decay_period
        self._action_space = action_space

    def get_action(self, t, observation, policy, **kwargs):
        action, agent_info = policy.get_action(observation)
        return self.get_action_from_raw_action(action, **kwargs), agent_info

    def get_action_from_raw_action(self, action, t=None, **kwargs):
        sigma = self._max_sigma - (self._max_sigma - self._min_sigma) * min(1.0, t * 1.0 / self._decay_period)
        return np.clip(action + np.random.normal(size=len(action)) * sigma, self._action_space.low,
                       self._action_space.high)
