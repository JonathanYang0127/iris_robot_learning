from collections import deque, OrderedDict
from functools import partial

import numpy as np

from rlkit.misc.eval_util import create_stats_ordered_dict
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.samplers.rollout_functions import rollout, fixed_contextual_rollout
from rlkit.exploration_strategies.embedding_wrappers import EmbeddingWrapperOffline, EmbeddingWrapper

def set_env_to_opp_task(env):
    if (isinstance(env, EmbeddingWrapper) or
        isinstance(env, EmbeddingWrapperOffline)):
        task_idx = env.env.task_idx
    else:
        task_idx = env.task_idx

    if env.is_reset_task():
        opp_task = task_idx - env.num_tasks
    else:
        opp_task = task_idx + env.num_tasks
    env.reset_task(opp_task)

class RndPathCollector(MdpPathCollector):
    def __init__(
            self,
            env,
            fwd_policy,
            perturb_policy,
            observation_keys=['observation',],
            epochs_per_reset=1,
            exploration_task=0,
            latent_dim=18,
            **kwargs
    ):
        super().__init__(env, fwd_policy, **kwargs)
        self._observation_keys = observation_keys
        self._reverse = False
        self._env = env
        self._fwd_policy = fwd_policy
        self._perturb_policy = perturb_policy
        self._epochs_per_reset = epochs_per_reset
        self._exploration_task = exploration_task
        self._epoch = 0
        self._latent_dim = latent_dim


    def collect_new_paths(
            self,
            *args,
            **kwargs
    ):
        def exploration_rollout(*args, **kwargs):
            # determine which task we're in
            self._reverse = self._env.is_reset_task()
            self._policy = self._perturb_policy if self._reverse else self._fwd_policy 

            # Should we really be assigning different embeddings
            # based on task direction here? We could use just one since 
            # we're training different policies. But the perturb policy may be trained on
            # fwd policy data.
            if self._reverse:
                embedding = np.zeros((self._latent_dim,))
                embedding[-1] = 1.0
            else:
                embedding = np.zeros((self._latent_dim,))
                embedding[-2] = 1.0
            print(embedding)
            rollout = fixed_contextual_rollout(*args,
                observation_keys=self._observation_keys,
                context=embedding,
                expl_reset_free=True, # expl_reset_free = true: no resets between episodes
                reverse=self._reverse,
                **kwargs)
            success = sum(rollout['rewards']) > 0
            print("reverse" if self._reverse else "forward", "success" if success else "failure", np.argmax(embedding))
            return rollout
        self._rollout_fn = exploration_rollout

        if self._epochs_per_reset != 0 and self._epoch % self._epochs_per_reset == 0:
            self._env.reset_task(self._exploration_task) 
            # alternate which task we reset to
            if (self._epoch // self._epochs_per_reset) % 2 == 0:
                set_env_to_opp_task(self._env)
            self._env.reset()
        
        paths = super().collect_new_paths(
            expl_reset_free=False, # expl_reset_free = false: always switch env task idx 
            *args, **kwargs)
        return paths

    def get_snapshot(self):
        snapshot = super().get_snapshot()
        snapshot.update(
            observation_keys=self._observation_keys,
        )
        return snapshot

    def end_epoch(self, epoch):
        self._epoch = epoch
        super().end_epoch(epoch)
