from collections import deque, OrderedDict
from functools import partial

import numpy as np

from rlkit.misc.eval_util import create_stats_ordered_dict
from rlkit.samplers.data_collector.base import PathCollector
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


class MdpPathCollector(PathCollector):
    def __init__(
            self,
            env,
            policy,
            max_num_epoch_paths_saved=None,
            rollout_fn=rollout,
            save_env_in_snapshot=True,
            **kwargs
    ):
        """

        :param env:
        :param policy:
        :param max_num_epoch_paths_saved: Maximum number of paths to save per
        epoch for computing statistics.
        :param rollout_fn: Some function with signature
        ```
        def rollout_fn(
            env, policy, max_path_length, *args, **kwargs
        ) -> List[Path]:
        ```

        :param save_env_in_snapshot: If True, save the environment in the
        snapshot.
        :param kwargs: Unused kwargs are passed on to `rollout_fn`
        """
        self._env = env
        self._policy = policy
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._rollout_fn = partial(rollout_fn, **kwargs)

        self._num_steps_total = 0
        self._num_paths_total = 0

        self._save_env_in_snapshot = save_env_in_snapshot

    def collect_new_paths(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths,
            object_detector=None,
            multi_task=False,
            task_index=0,
            expl_reset_free=False,
            log_obj_info=False,
            singletask_buffer=False,
    ):
        paths = []
        if log_obj_info:
            infos_list = []
        num_steps_collected = 0
        if multi_task and not expl_reset_free:
            self._env.reset_task(task_index)
        # print("num_steps", num_steps)

        while num_steps_collected < num_steps:
            # print("num_steps_collected", num_steps_collected)
            max_path_length_this_loop = min(  # Do not go over num_steps
                max_path_length,
                num_steps - num_steps_collected,
            )

            if object_detector is not None:
                input('Press enter when ready for online path rollout...')

            if log_obj_info:
                path, infos = self._rollout_fn(
                    self._env,
                    self._policy,
                    max_path_length=max_path_length_this_loop,
                )
            else:
                path = self._rollout_fn(
                    self._env,
                    self._policy,
                    max_path_length=max_path_length_this_loop,
                )

            if singletask_buffer and not expl_reset_free:
                self._env.reset()
            elif not expl_reset_free:
                # switch to opposite task
                set_env_to_opp_task(self._env)
            else:
                # switch to opposite task only if successful
                self._env.reset_robot_only()
                new_task_idx = self._env.get_new_task_idx()
                self._env.reset_task(new_task_idx)


            if object_detector is not None:
                from widowx_envs.scripts.label_pickplace_rewards import (
                    relabel_path_rewards_with_obj_model_and_thresh)
                path = relabel_path_rewards_with_obj_model_and_thresh(
                    object_detector, path, max_path_length_this_loop)

            path_len = len(path['actions'])
            if (
                    path_len != max_path_length
                    and not path['terminals'][-1]
                    and discard_incomplete_paths
            ):
                break
            num_steps_collected += path_len
            paths.append(path)
            if log_obj_info:
                infos_list.append(infos)
        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths)
        if log_obj_info:
            return paths, infos_list
        else:
            return paths

    def get_epoch_paths(self):
        return self._epoch_paths

    def end_epoch(self, epoch):
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)

    def get_diagnostics(self):
        path_lens = [len(path['actions']) for path in self._epoch_paths]
        stats = OrderedDict([
            ('num steps total', self._num_steps_total),
            ('num paths total', self._num_paths_total),
        ])
        stats.update(create_stats_ordered_dict(
            "path length",
            path_lens,
            always_show_all_stats=True,
        ))
        return stats

    def get_snapshot(self):
        snapshot_dict = dict(
            policy=self._policy,
        )
        if self._save_env_in_snapshot:
            snapshot_dict['env'] = self._env
        return snapshot_dict


class GoalConditionedPathCollector(MdpPathCollector):
    def __init__(
            self,
            *args,
            observation_key='observation',
            desired_goal_key='desired_goal',
            goal_sampling_mode=None,
            **kwargs
    ):
        def obs_processor(o):
            return np.hstack((o[observation_key], o[desired_goal_key]))

        rollout_fn = partial(
            rollout,
            preprocess_obs_for_policy_fn=obs_processor,
        )
        super().__init__(*args, rollout_fn=rollout_fn, **kwargs)
        self._observation_key = observation_key
        self._desired_goal_key = desired_goal_key
        self._goal_sampling_mode = goal_sampling_mode

    def collect_new_paths(self, *args, **kwargs):
        self._env.goal_sampling_mode = self._goal_sampling_mode
        return super().collect_new_paths(*args, **kwargs)

    def get_snapshot(self):
        snapshot = super().get_snapshot()
        snapshot.update(
            observation_key=self._observation_key,
            desired_goal_key=self._desired_goal_key,
        )
        return snapshot


class ObsDictPathCollector(MdpPathCollector):
    def __init__(
            self,
            *args,
            observation_keys=['observation',],
            **kwargs
    ):
        def obs_processor(obs):
            return np.concatenate([obs[key] for key in observation_keys])

        rollout_fn = partial(
            rollout,
            preprocess_obs_for_policy_fn=obs_processor,
        )
        super().__init__(*args, rollout_fn=rollout_fn, **kwargs)
        self._observation_keys = observation_keys

    def get_snapshot(self):
        snapshot = super().get_snapshot()
        snapshot.update(
            observation_keys=self._observation_keys,
        )
        return snapshot


class EmbeddingExplorationObsDictPathCollector(MdpPathCollector):
    def __init__(
            self,
            exploration_strategy,
            env,
            policy,
            observation_keys=['observation',],
            expl_reset_free=False,
            epochs_per_reset=1,
            exploration_task=0,
            do_cem_update=True,
            relabel_rewards=False,
            log_obj_info_path="",
            singletask_buffer=False, # if true, don't set to reverse task
            **kwargs
    ):
        '''
        Replaces context given by observation with context given by exploration strategy
        '''
        super().__init__(env, policy, **kwargs)
        self._exploration_strategy = exploration_strategy
        self._observation_keys = observation_keys
        self._expl_reset_free = expl_reset_free
        self._reverse = False
        self._env = env
        self._epochs_per_reset = epochs_per_reset
        self._exploration_task = exploration_task
        self._do_cem_update = do_cem_update
        self._relabel_rewards = relabel_rewards
        self._epoch = 0
        self.log_obj_info_path = log_obj_info_path
        self.log_obj_info = bool(log_obj_info_path)
        if self.log_obj_info:
            self.obj_infos_as_arr = None
        self.singletask_buffer = singletask_buffer

    def collect_new_paths(
            self,
            *args,
            **kwargs
    ):
        def exploration_rollout(*args, **kwargs):
            # determine which task we're in
            self._reverse = self._env.is_reset_task()
            # Dictionary of object_name -> np.array
            try:
                initial_obj_positions = self._env.env.get_obj_positions()
                target_object = self._env.env.target_object
                task_idx = self._env.env.task_idx
            except:
                initial_obj_positions = None
                target_object = None
                task_idx = None

            embedding = self._exploration_strategy.sample_embedding(reverse=self._reverse)
            rollout = fixed_contextual_rollout(*args,
                observation_keys=self._observation_keys,
                context=embedding,
                expl_reset_free=self._expl_reset_free,
                relabel_rewards=self._relabel_rewards,
                **kwargs)
            success = sum(rollout['rewards']) > 0
            post_trajectory_kwargs = {'reverse': self._reverse,
                'embedding': embedding,
                'success': success}
            print(post_trajectory_kwargs)
            if self._do_cem_update:
                self._exploration_strategy.post_trajectory_update(**post_trajectory_kwargs)
            if self.log_obj_info:
                return rollout, (initial_obj_positions, self._reverse, self._env.env.task_idx, target_object)
            else:
                return rollout
        self._rollout_fn = exploration_rollout

        if self._epochs_per_reset != 0 and self._epoch % self._epochs_per_reset == 0:
            self._env.reset_task(self._exploration_task)
            # alternate which task we reset to
            if (self._epoch // self._epochs_per_reset) % 2 == 0:
                set_env_to_opp_task(self._env)
            self._env.reset()

        if self.log_obj_info:
            paths, infos_list = super().collect_new_paths(
                expl_reset_free=self._expl_reset_free, log_obj_info=self.log_obj_info,
                singletask_buffer=self.singletask_buffer, *args, **kwargs)
            obj_infos_list = []
            for initial_obj_positions_dict, reverse, task_idx, target_object in infos_list:
                dict_to_add = {
                    "epoch": 1.0 * self._epoch, "reverse": 1.0 * reverse,
                    "task_idx": 1.0 * task_idx, "target_object_pos": initial_obj_positions_dict[target_object]
                }
                obj_infos_list.append(dict_to_add)

            def convert_dict_list_to_array(dict_list):
                """Converts the dict_list into an array and
                concatenates that to self.obj_infos_as_arr"""
                assert len(infos_list[0]) >= 1
                for _dict in dict_list:
                    values_as_list = []
                    for key, val in _dict.items():
                        # assuming that the dictionary ordering is fixed each time, which
                        # I think is true for python >=3.6
                        if isinstance(val, np.ndarray):
                            values_as_list.extend(list(val))
                        else:
                            values_as_list.append(float(val))
                    values_as_arr = np.array(values_as_list)[None]
                    if self.obj_infos_as_arr is not None:
                        self.obj_infos_as_arr = np.concatenate((self.obj_infos_as_arr, values_as_arr), axis=0)
                    else:
                        self.obj_infos_as_arr = values_as_arr

            convert_dict_list_to_array(obj_infos_list)
            np.save(self.log_obj_info_path, self.obj_infos_as_arr)
        else:
            paths = super().collect_new_paths(
                expl_reset_free=self._expl_reset_free, log_obj_info=self.log_obj_info,
                singletask_buffer=self.singletask_buffer, *args, **kwargs)
        return paths

    def get_snapshot(self):
        snapshot = super().get_snapshot()
        snapshot.update(
            observation_keys=self._observation_keys,
            exploration_strategy=self._exploration_strategy,
        )
        return snapshot

    def end_epoch(self, epoch):
        self._epoch = epoch
        super().end_epoch(epoch)

'''
class VAEWrappedEnvPathCollector(GoalConditionedPathCollector):
    def __init__(
            self,
            env: VAEWrappedEnv,
            policy,
            decode_goals=False,
            **kwargs
    ):
        super().__init__(env, policy, **kwargs)
        self._decode_goals = decode_goals

    def collect_new_paths(self, *args, **kwargs):
        self._env.decode_goals = self._decode_goals
        return super().collect_new_paths(*args, **kwargs)
'''
