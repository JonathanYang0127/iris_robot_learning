import warnings
from typing import Any, Callable, List

import numpy as np
from gym.spaces import Box, Dict
from multiworld.core.multitask_env import MultitaskEnv

from railrl import pythonplusplus as ppp
from railrl.core.distribution import DictDistribution
from railrl.envs.contextual import ContextualRewardFn
from railrl.envs.contextual.contextual_env import (
    ContextualDiagnosticsFn,
    Path,
    Context,
    Diagnostics,
)
from railrl.envs.images import Renderer

Goal = Any
GoalConditionedDiagnosticsFn = Callable[
    [List[Path], List[Goal]],
    Diagnostics,
]

class GoalDictDistributionFromMultitaskEnv(DictDistribution):
    def __init__(
            self,
            env: MultitaskEnv,
            desired_goal_keys=('desired_goal',),
    ):
        self._env = env
        self._desired_goal_keys = desired_goal_keys
        env_spaces = self._env.observation_space.spaces
        self._spaces = {
            k: env_spaces[k]
            for k in self._desired_goal_keys
        }

    def sample(self, batch_size: int):
        return {
            k: self._env.sample_goals(batch_size)[k]
            for k in self._desired_goal_keys
        }

    @property
    def spaces(self):
        return self._spaces


class AddImageDistribution(DictDistribution):
    def __init__(
            self,
            env: MultitaskEnv,
            base_distribution: DictDistribution,
            renderer: Renderer,
            image_goal_key='image_desired_goal',
            _suppress_warning=False,
    ):
        self._env = env
        self._base_distribution = base_distribution
        img_space = Box(0, 1, renderer.image_shape, dtype=np.float32)
        self._spaces = base_distribution.spaces
        self._spaces[image_goal_key] = img_space
        self._image_goal_key = image_goal_key
        self._renderer = renderer
        self._suppress_warning = _suppress_warning

    def sample(self, batch_size: int):
        if batch_size > 1 and not self._suppress_warning:
            warnings.warn(
                "Sampling many goals is slow. Consider using "
                "PresampledImageAndStateDistribution"
            )
        contexts = self._base_distribution.sample(batch_size)
        images = []
        for i in range(batch_size):
            goal = ppp.treemap(lambda x: x[i], contexts, atomic_type=np.ndarray)
            env_state = self._env.get_env_state()
            self._env.set_to_goal(goal)
            img_goal = self._renderer.create_image(self._env)
            self._env.set_env_state(env_state)
            images.append(img_goal)

        contexts[self._image_goal_key] = np.array(images)
        return contexts

    @property
    def spaces(self):
        return self._spaces


class PresampledDistribution(DictDistribution):
    def __init__(
            self,
            slow_sampler: DictDistribution,
            num_presampled_goals,
    ):
        self._sampler = slow_sampler
        self._num_presampled_goals = num_presampled_goals
        self._presampled_goals = self._sampler.sample(num_presampled_goals)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self._num_presampled_goals, batch_size)
        sampled_goals = {
            k: v[idx] for k, v in self._presampled_goals.items()
        }
        return sampled_goals

    @property
    def spaces(self):
        return self._sampler.spaces


class ContextualRewardFnFromMultitaskEnv(ContextualRewardFn):
    def __init__(
            self,
            env: MultitaskEnv,
            desired_goal_key='desired_goal',
            achieved_goal_key='achieved_goal',
            observation_key='observation',
    ):
        self._env = env
        self._desired_goal_key = desired_goal_key
        self._achieved_goal_key = achieved_goal_key
        self._observation_key = observation_key

    def __call__(self, states, actions, next_states, contexts):
        del states
        obs = {
           self._achieved_goal_key: next_states[self._observation_key],
           self._desired_goal_key: contexts[self._desired_goal_key],
        }
        return self._env.compute_rewards(actions, obs)


class GoalConditionedDiagnosticsToContextualDiagnostics(ContextualDiagnosticsFn):
    # use a class rather than function for serialization
    def __init__(
            self,
            goal_conditioned_diagnostics: GoalConditionedDiagnosticsFn,
            goal_key: str
    ):
        self._goal_conditioned_diagnostics = goal_conditioned_diagnostics
        self._goal_key = goal_key

    def __call__(self, paths: List[Path], contexts: List[Context]) -> Diagnostics:
        goals = [c[self._goal_key] for c in contexts]
        return self._goal_conditioned_diagnostics(paths, goals)
