from functools import partial

from rlkit.policies.base import Policy
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.samplers.rollout_functions import contextual_rollout
from rlkit.torch.pearl.sampler import rollout
from rlkit.torch.pearl.agent import PEARLAgent


class PearlPathCollector(MdpPathCollector):
    def __init__(
            self,
            env: PEARLAgent,
            policy: Policy,
            **kwargs
    ):
        rollout_fn = rollout
        super().__init__(
            env, policy, rollout_fn=rollout_fn,
            **kwargs
        )

    def collect_new_steps(
            self,
            task_idx,
            *args,
            **kwargs
            # num_steps,
            # discard_incomplete_paths,
    ):
        self.env.reset_task(task_idx)
        return super().collect_new_paths(*args, **kwargs)
