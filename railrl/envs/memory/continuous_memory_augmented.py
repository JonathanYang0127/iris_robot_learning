import numpy as np

from railrl.misc.rllab_util import split_flat_product_space_into_components_n
from rllab.envs.base import Env
from rllab.envs.proxy_env import ProxyEnv
from rllab.misc.overrides import overrides
from rllab.spaces.product import Product
from rllab.spaces.box import Box


class ContinuousMemoryAugmented(ProxyEnv):
    """
    An environment that wraps another environments and adds continuous memory
    states/actions.
    """
    def __init__(
            self,
            env: Env,
            num_memory_states=10
    ):
        super().__init__(env)
        self._num_memory_states = num_memory_states
        self._memory_state = np.zeros(self._num_memory_states)
        self._action_space = Product(
            env.action_space,
            self._memory_state_space()
        )
        self._observation_space = Product(
            env.observation_space,
            self._memory_state_space()
        )

    def _memory_state_space(self):
        return Box(-np.ones(self._num_memory_states),
                   np.ones(self._num_memory_states))

    def reset(self):
        self._memory_state = np.zeros(self._num_memory_states)
        env_obs = self._wrapped_env.reset()
        return env_obs, self._memory_state

    def step(self, action):
        """
        :param action: An unflattened action, i.e. action = (environment
        action,
        memory write) tuple.
        :return: An unflattened observation.
        """
        env_action, memory_state = action
        observation, reward, done, info = self._wrapped_env.step(env_action)
        return (
            # Squeeze the memory state since the returned next_observation
            # should be flat.
            (observation, memory_state.squeeze()),
            reward,
            done,
            info
        )

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def memory_dim(self):
        return self._num_memory_states

    def _strip_path(self, path):
        path = path.copy()
        actions = path['actions']
        env_actions = split_flat_product_space_into_components_n(
            self.action_space,
            actions
        )[0]
        path['actions'] = env_actions

        observations = path['observations']
        env_obs = split_flat_product_space_into_components_n(
            self.observation_space,
            observations
        )[0]
        path['observations'] = env_obs

        return path

    def log_diagnostics(self, paths):
        non_memory_paths = [self._strip_path(path) for path in paths]
        return self._wrapped_env.log_diagnostics(non_memory_paths)