from rllab.exploration_strategies.base import ExplorationStrategy
import numpy as np
from railrl.misc.np_util import to_onehot


class OneHotSampler(ExplorationStrategy):
    """
    Given a probability distribution over a set of discrete action, this ES
    samples one value and returns a one-hot vector.
    """
    def get_action(self, t, observation, policy, **kwargs):
        action, _ = policy.get_action(observation)
        return self.get_action_from_raw_action(action)

    def get_action_from_raw_action(self, action, **kwargs):
        num_values = len(action)
        elements = np.arange(num_values)
        number = np.random.choice(elements, p=action)
        return to_onehot(number, num_values)