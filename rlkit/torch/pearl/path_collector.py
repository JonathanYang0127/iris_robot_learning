import numpy as np

from rlkit.policies.base import Policy
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.pearl.sampler import rollout
from rlkit.torch.pearl.agent import PEARLAgent


class PearlPathCollector(MdpPathCollector):
    def __init__(
            self,
            env: PEARLAgent,
            policy: Policy,
            task_indices,
            rollout_fn=rollout,
            **kwargs
    ):
        super().__init__(
            env, policy,
            rollout_fn=rollout_fn,
            **kwargs
        )
        self.task_indices = task_indices

    def collect_new_paths(
            self,
            *args,
            task_idx=None,
            **kwargs
    ):
        task_idx = task_idx or np.random.choice(self.task_indices)
        self._env.reset_task(task_idx)
        return super().collect_new_paths(*args, **kwargs)
