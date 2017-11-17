from collections import OrderedDict

import numpy as np
from gym.envs.mujoco import PusherEnv as GymPusherEnv

from railrl.misc.data_processing import create_stats_ordered_dict
from railrl.misc.rllab_util import get_stat_in_dict
from rllab.misc import logger


class PusherEnv(GymPusherEnv):
    def reset_model(self):
        qpos = self.init_qpos

        while True:
            self.cylinder_pos = np.concatenate([
                self.np_random.uniform(low=-0.3, high=0, size=1),
                self.np_random.uniform(low=-0.2, high=0.2, size=1)])
            if np.linalg.norm(self.cylinder_pos - self.goal_cylinder_xy) > 0.17:
                break

        qpos[-4:-2] = self.cylinder_pos
        qpos[-2:] = self.goal_cylinder_xy
        qvel = self.init_qvel + self.np_random.uniform(low=-0.005,
                                                       high=0.005,
                                                       size=self.model.nv)
        qvel[-4:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _step(self, a):
        arm_to_object = (
            self.get_body_com("tips_arm") - self.get_body_com("object")
        )
        object_to_goal = (
            self.get_body_com("object") - self.get_body_com("goal")
        )
        arm_to_goal = (
            self.get_body_com("tips_arm") - self.get_body_com("goal")
        )
        obs, reward, done, info_dict = super()._step(a)
        info_dict['arm to object distance'] = np.linalg.norm(arm_to_object)
        info_dict['object to goal distance'] = np.linalg.norm(object_to_goal)
        info_dict['arm to goal distance'] = np.linalg.norm(arm_to_goal)
        return obs, reward, done, info_dict

    def log_diagnostics(self, paths):
        statistics = OrderedDict()

        for stat_name in [
            'arm to object distance',
            'object to goal distance',
            'arm to goal distance',
        ]:
            stat = get_stat_in_dict(
                paths, 'env_infos', stat_name
            )
            statistics.update(create_stats_ordered_dict(
                stat_name, stat
            ))

        for key, value in statistics.items():
            logger.record_tabular(key, value)

    @property
    def goal_cylinder_xy(self):
        return np.array([0, 0])