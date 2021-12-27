import abc
from collections import OrderedDict

import gym
import gym.spaces
import numpy as np
from typing import Union, Callable, Any, Dict, List

from rlkit.core.distribution import DictDistribution
from rlkit.torch import pytorch_util as ptu
from rlkit.util.io import load_local_or_remote_file
from rlkit import pythonplusplus as ppp


Path = Dict
Diagnostics = Dict
Context = Any
ContextualDiagnosticsFn = Callable[
    [List[Path], List[Context]],
    Diagnostics,
]


def batchify(x):
    return ppp.treemap(lambda x: x[None], x, atomic_type=np.ndarray)


def insert_reward(contexutal_env, info, obs, reward, done):
    info['ContextualEnv/old_reward'] = reward
    return info


def delete_info(contexutal_env, info, obs, reward, done):
    return {}


def maybe_flatten_obs(self, obs):
    if len(obs.shape) == 1:
        return obs.reshape(1, -1)
    return obs


class ContextualRewardFn(object, metaclass=abc.ABCMeta):
    """You can also just pass in a function."""

    @abc.abstractmethod
    def __call__(
            self,
            states: dict,
            actions,
            next_states: dict,
            contexts: dict
    ):
        pass


class UnbatchRewardFn(object):
    def __init__(self, reward_fn: ContextualRewardFn):
        self._reward_fn = reward_fn

    def __call__(
            self,
            state: dict,
            action,
            next_state: dict,
            context: dict
    ):
        states = batchify(state)
        actions = batchify(action)
        next_states = batchify(next_state)
        return self._reward_fn(
            states,
            actions,
            next_states,
            context,
        )[0]


class ContextualEnv(gym.Wrapper):

    def __init__(
            self,
            env: gym.Env,
            context_distribution: DictDistribution,
            reward_fn: ContextualRewardFn,
            observation_key=None,  # for backwards compatibility
            observation_keys=None,
            update_env_info_fn=None,
            contextual_diagnostics_fns: Union[
                None, List[ContextualDiagnosticsFn]] = None,
            unbatched_reward_fn=None,
    ):
        super().__init__(env)

        if observation_key is not None and observation_keys is not None:
            raise ValueError(
                'Only specify observation_key or observation_keys')

        if observation_key is None and observation_keys is None:
            raise ValueError(
                'Specify either observation_key or observation_keys'
            )

        if observation_keys is None:
            observation_keys = [observation_key]

        if contextual_diagnostics_fns is None:
            contextual_diagnostics_fns = []

        if not isinstance(env.observation_space, gym.spaces.Dict):
            raise ValueError("ContextualEnvs require wrapping Dict spaces.")

        spaces = env.observation_space.spaces
        for key, space in context_distribution.spaces.items():
            spaces[key] = space

        self.observation_space = gym.spaces.Dict(spaces)
        self.reward_fn = reward_fn
        self._observation_keys = observation_keys
        self._last_obs = None
        self._update_env_info = update_env_info_fn or insert_reward

        self._rollout_context_batch = None

        self.context_distribution = context_distribution
        self._context_keys = list(context_distribution.spaces.keys())

        self._contextual_diagnostics_fns = contextual_diagnostics_fns

        if unbatched_reward_fn is None:
            unbatched_reward_fn = UnbatchRewardFn(reward_fn)

        self.unbatched_reward_fn = unbatched_reward_fn

    def reset(self):
        obs = self.env.reset()
        self._rollout_context_batch = self.context_distribution(
            context=obs).sample(1)
        self._add_context_to_obs(obs)
        self._last_obs = obs
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._add_context_to_obs(obs)
        new_reward = self._compute_reward(self._last_obs, action, obs, reward)
        self._last_obs = obs
        info = self._update_env_info(self, info, obs, reward, done)
        return obs, new_reward, done, info

    def _compute_reward(self, state, action, next_state, env_reward=None):
        """Do reshaping for reward_fn, which is implemented for batches."""
        # TODO: don't assume these things are just vectors
        if not self.reward_fn:
            return env_reward
        else:
            return self.unbatched_reward_fn(
                state, action, next_state, self._rollout_context_batch)

    def _add_context_to_obs(self, obs):
        for key in self._context_keys:
            obs[key] = self._rollout_context_batch[key][0]

    def get_diagnostics(self, paths):
        stats = OrderedDict()
        contexts = [self._get_context(p) for p in paths]
        for fn in self._contextual_diagnostics_fns:
            stats.update(fn(paths, contexts))
        return stats

    def _get_context(self, path):
        first_observation = path['observations'][0]
        return {
            key: first_observation[key] for key in self._context_keys
        }


class PlannedContextualEnv(gym.Wrapper):

    def __init__(
            self,
            env: gym.Env,
            planner,
            reward_fn: ContextualRewardFn,
            presampled_data_path,
            context_timeout=None,
            direction_timeout=None,
            context_switch_reward_thresh=None,
            observation_key=None,  # for backwards compatibility
            observation_keys=None,
            update_env_info_fn=None,
            contextual_diagnostics_fns: Union[
                None, List[ContextualDiagnosticsFn]] = None,
            unbatched_reward_fn=None,
    ):
        super().__init__(env)

        if observation_key is not None and observation_keys is not None:
            raise ValueError(
                'Only specify observation_key or observation_keys')

        if observation_key is None and observation_keys is None:
            raise ValueError(
                'Specify either observation_key or observation_keys'
            )

        if observation_keys is None:
            observation_keys = [observation_key]

        if contextual_diagnostics_fns is None:
            contextual_diagnostics_fns = []

        if not isinstance(env.observation_space, gym.spaces.Dict):
            raise ValueError("ContextualEnvs require wrapping Dict spaces.")

        spaces = env.observation_space.spaces
        for key, space in planner.spaces.items():
            spaces[key] = space

        self.observation_space = gym.spaces.Dict(spaces)
        self.reward_fn = reward_fn
        self._observation_keys = observation_keys

        self._last_obs = None
        self._update_env_info = update_env_info_fn or insert_reward

        # presampled_data = np.load(presampled_data_path, allow_pickle=True)
        presampled_data = load_local_or_remote_file(presampled_data_path)

        self._init_images = presampled_data['initial_image_observation']
        self._goal_images = presampled_data['image_desired_goal']

        self._init_latent_states = self._encode_images(
            self._preprocess_images(self._init_images))
        self._goal_latent_states = self._encode_images(
            self._preprocess_images(self._goal_images))

        self._planner = planner
        self._context_keys = list(planner.spaces.keys())
        self._contextual_diagnostics_fns = contextual_diagnostics_fns
        self._rollout_context_batch = None

        self._context_timeout = context_timeout
        self._direction_timeout = direction_timeout

        self._context_switch_reward_thresh = context_switch_reward_thresh

        if unbatched_reward_fn is None:
            unbatched_reward_fn = UnbatchRewardFn(reward_fn)

        self.unbatched_reward_fn = unbatched_reward_fn

    def _preprocess_images(self, images):
        num_samples = images.shape[0]
        num_steps = images.shape[1]
        images = np.reshape(
            images, [num_samples, num_steps, 3, 48, 48])
        images = np.transpose(images, [0, 1, 4, 3, 2])

        images = images.astype(np.float32) / 255 - 0.5

        # Channel first.
        if images.ndim == 4:
            images = np.transpose(images, (0, 3, 1, 2))
        elif images.ndim == 5:
            images = np.transpose(images, (0, 1, 4, 2, 3))

        return images

    def _encode_images(self, images, batch_size=1024):
        num_seqs = images.shape[0]
        num_steps = images.shape[1]
        num_samples = num_seqs * num_steps
        num_batches = int(np.ceil(float(num_samples) / float(batch_size)))
        encodings = []
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            end = min(end, num_samples)

            batch = images[start:end]
            batch = ptu.from_numpy(batch)
            batch = batch.view(
                -1, batch.shape[-3], batch.shape[-2], batch.shape[-1])

            encoding_i = self.model.encode(batch, mode='zq', flatten=False)
            encoding_i = ptu.get_numpy(encoding_i)
            encodings.append(encoding_i)
            # logging.info('Finished encoding the images %d / %d.'
            #              % (end, num_samples))

        encodings = np.concatenate(encodings, axis=0)

        return encodings

    def reset(self):
        obs = self.env.reset()
        self._replan(obs, 1)
        self._switch_context()
        self._add_context_to_obs(obs)
        self._last_obs = obs
        return obs

    def step(self, action):
        last_obs = self._last_obs
        obs, reward, done, info = self.env.step(action)
        self._add_context_to_obs(obs)
        new_reward = self._compute_reward(self._last_obs, action, obs, reward)

        self._countdown_this_context = max(
            0, self._countdown_this_context - 1)
        self._countdown_this_direction = max(
            0, self._countdown_this_direction - 1)

        # Maybe replan.
        if self._countdown_this_direction == 0:
            direction = 1 - self._direction  # Reverse the direction.
            self._replan(obs, direction)

        # Maybe switch to the next context.
        if (self._is_context_reached(last_obs, action, obs, reward) or
                self._countdown_this_context == 0):
            # TODO: Check whthere there are still remaining goals.
            self._switch_context()

        self._last_obs = obs
        info = self._update_env_info(self, info, obs, reward, done)
        return obs, new_reward, done, info

    def _replan(self, obs, direction):
        if direction == 0:
            goal_states = self._init_latent_states[
                self.reward_fn.observation_key]
        elif direction == 1:
            goal_states = self._goal_latent_states[
                self.reward_fn.observation_key]
        else:
            raise ValueError('Unrecognized direction: %r' % (direction))

        curr_image = obs['image_observation']
        curr_state = self.model.encode(curr_image, mode='zq', flatten=False)
        self._rollout_context_batch = self.planner(
            init_state=curr_state,
            goal_state=goal_states)  # NOQA

        if self._direction_timeout is not None:
            self._countdown_this_direction = self._direction_timeout
        else:
            self._countdown_this_direction = 0

        self._direction = direction

    def _switch_context(self):
        if self._context_timeout is not None:
            self._countdown_this_context = self._context_timeout
        else:
            self._countdown_this_context = 0

        self._current_context = {}
        for key in self._context_keys:
            # TODO(kuanfang)
            self._current_context[key] = self._context_path[key][0]
            self._context_path[key] = self._context_path[key][1:]

    def _is_context_reached(self, last_obs, action, obs, reward):
        if self._context_switch_reward_thresh is None:
            return False

        assert self.reward_fn is not None
        s = self.reward_fn.process(
            obs[self.reward_fn.observation_key])
        c = self.reward_fn.process(
            self._rollout_context_batch[self.reward_fn.desired_goal_key])

        if np.linalg.norm(s - c, axis=1) < self._context_switch_reward_thresh:
            return True
        else:
            return False

    def _add_context_to_obs(self, obs):
        for key in self._context_keys:
            obs[key] = self._rollout_context_batch[key][0]

    def _compute_reward(self, state, action, next_state, env_reward=None):
        """Do reshaping for reward_fn, which is implemented for batches."""
        # TODO: don't assume these things are just vectors
        if not self.reward_fn:
            return env_reward
        else:
            return self.unbatched_reward_fn(
                state, action, next_state, self._rollout_context_batch)

    def get_diagnostics(self, paths):
        stats = OrderedDict()
        contexts = [self._get_context(p) for p in paths]
        for fn in self._contextual_diagnostics_fns:
            stats.update(fn(paths, contexts))
        return stats

    def _get_context(self, path):
        first_observation = path['observations'][0]
        return {
            key: first_observation[key] for key in self._context_keys
        }
