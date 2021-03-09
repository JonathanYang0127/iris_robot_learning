from typing import Union

import numpy as np
from gym import Env

from rlkit.data_management.multitask_replay_buffer import MultiTaskReplayBuffer
from rlkit.policies.base import Policy
import rlkit.torch.pytorch_util as ptu
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.samplers.data_collector.joint_path_collector import \
    JointPathCollector
from rlkit.torch.pearl.buffer import PearlReplayBuffer
from rlkit.torch.pearl.sampler import (
    rollout, rollout_multiple_and_flatten,
    rollout_multiple,
    merge_paths,
)
from rlkit.torch.pearl.agent import PEARLAgent, MakePEARLAgentDeterministic
import math


class PearlPathCollector(MdpPathCollector):
    def __init__(
            self,
            env: Env,
            policy: Union[PEARLAgent, MakePEARLAgentDeterministic],
            task_indices,
            replay_buffer: PearlReplayBuffer,
            rollout_fn=rollout,
            sample_initial_context=False,
            accum_context_across_rollouts=False,
            **kwargs
    ):
        super().__init__(
            env, policy,
            rollout_fn=rollout_fn,
            **kwargs
        )
        self.replay_buffer = replay_buffer
        self.task_indices = task_indices
        self._rollout_kwargs = kwargs
        self._sample_initial_context = sample_initial_context
        self.accum_context_across_rollouts = accum_context_across_rollouts

    def collect_new_paths_and_indices(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths,
            initial_context=None,
            task_idx=None,
            task_indices_for_rollout=None,
            **kwargs
    ):
        """

        :param max_path_length:
        :param num_steps:
        :param discard_incomplete_paths:
        :param task_idx: If set, only collect trajectories with this task index.
        :param task_indices_for_rollout: If this is set and `task_idx` is not set, then cycle through these task indices at each rollout. If neither `task_indices_for_rollout` nor `task_idx` are set, then randomly sample task indices.
        :return:
        """
        paths = []
        task_indices = []
        num_steps_collected = 0
        # original_task_idx = task_idx
        if task_idx:
            task_indices_for_rollout = [task_idx]
        # if self._sample_initial_context and original_task_idx is not None:
        #     # TODO: fix hack and consolidate where init context is sampled
        #     try:
        #         initial_context = self.replay_buffer.sample_context(original_task_idx)
        #         initial_context = ptu.from_numpy(initial_context)
        #     except ValueError:
        #         # this is needed for just the first loop where we need to fill the replay buffer without setting the replay buffer
        #         initial_context = None
        # else:
        #     initial_context = None
        init_context_this_loop = initial_context
        loop_i = 0
        while num_steps_collected < num_steps:
            if task_indices_for_rollout is None:
                task_idx_this_loop = np.random.choice(self.task_indices)
            else:
                task_idx_this_loop = task_indices_for_rollout[
                    loop_i % len(task_indices_for_rollout)
                ]
            task_indices.append(task_idx_this_loop)
            max_path_length_this_loop = min(  # Do not go over num_steps
                max_path_length,
                num_steps - num_steps_collected,
            )

            if init_context_this_loop is None:
                init_context_this_loop = self._get_initial_context(task_idx_this_loop)
            path = self._rollout_fn(
                self._env,
                self._policy,
                max_path_length=max_path_length_this_loop,
                initial_context=init_context_this_loop,
                task_idx=task_idx_this_loop,
                **kwargs
            )
            if self.accum_context_across_rollouts:
                init_context_this_loop = path['context']
            else:
                init_context_this_loop = None
            path_len = len(path['actions'])
            if (
                    path_len != max_path_length
                    and not path['terminals'][1]
                    and discard_incomplete_paths
            ):
                continue
            num_steps_collected += path_len
            paths.append(path)
            loop_i += 1
        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths)
        return paths, task_indices

    def _get_initial_context(self, task_idx):
        if self._sample_initial_context:
            try:
                initial_context = self.replay_buffer.sample_context(task_idx)
                initial_context = ptu.from_numpy(initial_context)
            except ValueError:
                # this is needed for just the first loop where we need to fill the replay buffer without setting the replay buffer
                # TODO: fix hack
                initial_context = None
        else:
            initial_context = None
        return initial_context

    def get_snapshot(self):
        snapshot = super().get_snapshot()
        snapshot.update(
            rollout_kwargs=self._rollout_kwargs,
            task_indices=self.task_indices,
        )
        return snapshot

    def collect_new_paths(self, max_path_length, num_steps,
                          discard_incomplete_paths,
                          **kwargs):
        return self.collect_new_paths_and_indices(
            max_path_length=max_path_length,
            num_steps=num_steps,
            discard_incomplete_paths=discard_incomplete_paths,
            **kwargs
        )[0]


class PearlMultiPathCollector(MdpPathCollector):
    """This path collector collects multiple paths at once. This is useful
    to test online adaption where the context/task needs to stay consistent
    across episodes."""
    def __init__(
            self,
            env: PEARLAgent,
            policy: Policy,
            task_indices,
            replay_buffer: PearlReplayBuffer,
            sample_initial_context=False,
            n_repeats=3,
            **kwargs
    ):
        super().__init__(
            env, policy,
            rollout_fn=rollout_multiple,
            **kwargs
        )
        self.replay_buffer = replay_buffer
        self._rollout_kwargs = kwargs
        self._sample_initial_context = sample_initial_context
        self.n_repeats = n_repeats
        self.task_indices = task_indices
        self._num_path_batches_total = 0

    def collect_new_paths_and_indices(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths,
            task_idx=None,
            task_indices_for_rollout=None,
            flatten_paths=False,
    ):
        """

        :param max_path_length:
        :param num_steps:
        :param discard_incomplete_paths: ignored
        :param task_idx: If set, only collect trajectories with this task index.
        :param task_indices_for_rollout: If this is set and `task_idx` is not set, then cycle through these task indices at each batch of rollouts. If neither `task_indices_for_rollout` nor `task_idx` are set, then randomly sample task indices.
        :return:
        """
        if task_idx:
            task_indices_for_rollout = [task_idx]
        del task_idx
        del discard_incomplete_paths
        all_paths = []
        task_indices = []
        num_steps_collected = 0
        loop_i = 0
        while num_steps_collected < num_steps:
            if task_indices_for_rollout is None:
                task_idx_this_loop = np.random.choice(self.task_indices)
            else:
                task_idx_this_loop = task_indices_for_rollout[
                    loop_i % len(task_indices_for_rollout)
                    ]
            task_indices.append(task_idx_this_loop)
            max_path_length_this_loop = min(  # Do not go over num_steps
                max_path_length,
                num_steps - num_steps_collected,
            )
            n_repeats = min(
                self.n_repeats,
                int(math.ceil(
                    (num_steps - num_steps_collected) / max_path_length
                )),
            )
            # if n_repeats == 0:
            #     break
            init_context_this_loop = self._get_initial_context(task_idx_this_loop)
            paths = rollout_multiple(
                env=self._env,
                agent=self._policy,
                task_idx=task_idx_this_loop,
                initial_context=init_context_this_loop,
                max_path_length=max_path_length_this_loop,
                num_repeats=n_repeats,
            )
            num_steps_collected += sum(len(path['actions']) for path in paths)
            self._num_steps_total += num_steps_collected
            self._num_paths_total += len(paths)
            self._num_path_batches_total += 1
            self._epoch_paths.extend(paths)
            if flatten_paths:
                all_paths.append(merge_paths(paths))
            else:
                all_paths += paths
            loop_i += 1
        return all_paths, task_indices

    def _get_initial_context(self, task_idx):
        if self._sample_initial_context:
            initial_context = self.replay_buffer.sample_context(task_idx)
            # Note: returns None if there's nothing in the buffer
        else:
            initial_context = None
        if initial_context is not None:
            initial_context = ptu.from_numpy(initial_context)
        return initial_context

    def get_snapshot(self):
        snapshot = super().get_snapshot()
        snapshot.update(
            rollout_kwargs=self._rollout_kwargs,
        )
        return snapshot

    def collect_new_paths(self, max_path_length, num_steps,
                          discard_incomplete_paths,
                          **kwargs):
        raise NotImplementedError()


class PearlJointPathCollector(JointPathCollector):
    def collect_new_paths_and_indices(self, *args, **kwargs):
        all_paths, all_indices = [], []
        for paths, indices in self.yield_new_paths_and_indices(*args, **kwargs):
            all_paths += paths
            all_indices += indices
        return all_paths, all_indices

    def yield_new_paths_and_indices(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths,
            **kwargs
    ):
        for (
                name, paths_this_loop, tasks_indices_this_loop
        ) in self.yield_name_paths_and_indices(
                max_path_length,
                num_steps,
                discard_incomplete_paths,
                **kwargs
        ):
            yield paths_this_loop, tasks_indices_this_loop

    def yield_name_paths_and_indices(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths,
            per_name_callback=None,
            **kwargs
    ):
        for name, collector in self.path_collectors.items():
            if per_name_callback:
                kwargs = per_name_callback(name, kwargs)
            paths, indices = collector.collect_new_paths_and_indices(
                max_path_length=max_path_length,
                num_steps=self._get_num_steps(num_steps, name),
                discard_incomplete_paths=discard_incomplete_paths,
                **kwargs
            )
            yield name, paths, indices

    def collect_new_paths(self, *args, **kwargs):
        return self.collect_new_paths_and_indices(*args, **kwargs)[0]
