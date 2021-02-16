import numpy as np

from rlkit.policies.base import Policy
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.pearl.buffer import PearlReplayBuffer
from rlkit.torch.pearl.sampler import rollout
from rlkit.torch.pearl.agent import PEARLAgent


class PearlPathCollector(MdpPathCollector):
    def __init__(
            self,
            env: PEARLAgent,
            policy: Policy,
            task_indices,
            replay_buffer: PearlReplayBuffer,
            rollout_fn=rollout,
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

    def collect_new_paths(
            self,
            *args,
            task_idx=None,
            initial_context=None,
            **kwargs
    ):
        task_idx = task_idx or np.random.choice(self.task_indices)
        if initial_context is None:
            # TODO: fix hack and consolidate where init context is sampled
            try:
                initial_context = self.replay_buffer.sample_context(task_idx )
            except ValueError:
                # this is needed for just the first loop where we need to fill the replay buffer without setting the replay buffer
                pass
        self._env.reset_task(task_idx)
        return super().collect_new_paths(*args, initial_context=initial_context, **kwargs)

    def get_snapshot(self):
        snapshot = super().get_snapshot()
        snapshot.update(
            rollout_kwargs=self._rollout_kwargs,
            task_indices=self.task_indices,
        )
        return snapshot
