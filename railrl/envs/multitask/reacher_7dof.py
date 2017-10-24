from collections import OrderedDict

import torch
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

from railrl.envs.env_utils import get_asset_xml
from railrl.envs.multitask.multitask_env import MultitaskEnv
from railrl.misc.data_processing import create_stats_ordered_dict
from railrl.misc.rllab_util import get_stat_in_dict
import railrl.torch.pytorch_util as ptu
from rllab.misc import logger


class Reacher7DofMultitaskEnv(
    MultitaskEnv, mujoco_env.MujocoEnv, utils.EzPickle
):
    def __init__(self):
        MultitaskEnv.__init__(self)
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(
            self,
            get_asset_xml('reacher_7dof.xml'),
            5,
        )
        self._desired_xyz = np.zeros(3)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 4.0

    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel + self.np_random.uniform(low=-0.005,
                                                       high=0.005, size=self.model.nv)
        qpos[-7:-4] = self._desired_xyz
        qvel[-7:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def sample_actions(self, batch_size):
        return np.random.uniform(
            -1, 1, size=(batch_size, self.action_space.low.size)
        )

    def sample_states(self, batch_size):
        return np.hstack((
            # From the xml
            self.np_random.uniform(low=-2.28, high=1.71, size=(batch_size, 1)),
            self.np_random.uniform(low=-0.52, high=1.39, size=(batch_size, 1)),
            self.np_random.uniform(low=-1.4, high=1.7, size=(batch_size, 1)),
            self.np_random.uniform(low=-2.32, high=0, size=(batch_size, 1)),
            self.np_random.uniform(low=-1.5, high=1.5, size=(batch_size, 1)),
            self.np_random.uniform(low=-1.094, high=0, size=(batch_size, 1)),
            self.np_random.uniform(low=-1.5, high=1.5, size=(batch_size, 1)),
            # velocities
            self.np_random.uniform(low=-1, high=1, size=(batch_size, 7)),
        ))

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[:7],
            self.model.data.qvel.flat[:7],
            self.get_body_com("tips_arm"),
        ])

    def _step(self, a):
        distance = np.linalg.norm(
            self.get_body_com("tips_arm") - self.get_body_com("goal")
        )
        reward = - distance
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(distance=distance)

    def log_diagnostics(self, paths):
        super().log_diagnostics(paths)
        statistics = OrderedDict()

        euclidean_distances = get_stat_in_dict(
            paths, 'env_infos', 'distance'
        )
        statistics.update(create_stats_ordered_dict(
            'Euclidean distance to goal', euclidean_distances
        ))
        statistics.update(create_stats_ordered_dict(
            'Final Euclidean distance to goal',
            euclidean_distances[:, -1],
            always_show_all_stats=True,
        ))
        for key, value in statistics.items():
            logger.record_tabular(key, value)


class Reacher7DofXyzGoalState(Reacher7DofMultitaskEnv):
    """
    The goal state is just the XYZ location of the end effector.
    """
    def sample_goal_states(self, batch_size):
        # Number taken from running a random policy and seeing what XYZ values
        # are reached
        return np.hstack((
            self.np_random.uniform(low=-0.75, high=0.75, size=(batch_size, 1)),
            self.np_random.uniform(low=-1.25, high=0.25, size=(batch_size, 1)),
            self.np_random.uniform(low=-0.2, high=0.6, size=(batch_size, 1)),
        ))

    def set_goal(self, goal):
        super().set_goal(goal)
        self._desired_xyz = goal

    @property
    def goal_dim(self):
        return 3

    def convert_obs_to_goal_states(self, obs):
        return obs[:, -3:]


class Reacher7DofAngleGoalState(Reacher7DofMultitaskEnv):
    """
    The goal state is joint angles
    """
    def sample_goal_states(self, batch_size):
        # Number taken from running a random policy and seeing what XYZ values
        # are reached
        states = super().sample_states(batch_size)
        return states[:, :7]

    @property
    def goal_dim(self):
        return 7

    def convert_obs_to_goal_states(self, obs):
        return obs[:, :7]

    def set_goal(self, goal):
        super().set_goal(goal)

        saved_qpos = self.init_qpos.copy()
        saved_qvel = self.init_qvel.copy()
        qpos_tmp = saved_qpos.copy()
        qpos_tmp[:7] = goal
        self.set_state(qpos_tmp, saved_qvel)
        self._desired_xyz = self.get_body_com("tips_arm")
        saved_qpos[7:10] = self._desired_xyz
        self.set_state(saved_qpos, saved_qvel)


class Reacher7DofFullGoalState(Reacher7DofMultitaskEnv):
    """
    The goal state is the full state: joint angles and velocities.
    """
    def modify_goal_state_for_rollout(self, goal_state):
        # set desired velocity to zero
        goal_state[-7:] = 0
        return goal_state

    def sample_goal_states(self, batch_size):
        return self.sample_states(batch_size)

    def sample_irrelevant_goal_dimensions(self, goal, batch_size):
        new_goal_states = super().sample_irrelevant_goal_dimensions(
            goal, batch_size
        )
        new_goal_states[:, -7:] = (
            self.np_random.uniform(low=-1, high=1, size=(batch_size, 7))
        )
        return new_goal_states

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[:7],
            self.model.data.qvel.flat[:7],
        ])

    @property
    def goal_dim(self):
        return 14

    def set_goal(self, goal):
        """
        I don't want to manually compute the forward dynamics to compute the
        goal state XYZ coordinate.

        Instead, I just put the arm in the desired goal state. Then I measure
        the end-effector XYZ coordinate.
        """
        super().set_goal(goal)

        saved_qpos = self.init_qpos.copy()
        saved_qvel = self.init_qvel.copy()
        qpos_tmp = saved_qpos.copy()
        qpos_tmp[:7] = goal[:7]
        self.set_state(qpos_tmp, saved_qvel)
        self._desired_xyz = self.get_body_com("tips_arm")
        saved_qpos[7:10] = self._desired_xyz
        self.set_state(saved_qpos, saved_qvel)

    def log_diagnostics(self, paths):
        super().log_diagnostics(paths)
        statistics = OrderedDict()

        observations = np.vstack([path['observations'] for path in paths])
        goal_states = np.vstack([path['goal_states'] for path in paths])
        differences = observations - goal_states
        angle_differences = differences[:, :7]
        vel_differences = differences[:, 7:]
        for l in [1, 2]:
            statistics.update(create_stats_ordered_dict(
                'L{} full goal distance to target'.format(l),
                np.linalg.norm(differences, axis=1, ord=l)
            ))
            statistics.update(create_stats_ordered_dict(
                'L{} angle goal distance to target'.format(l),
                np.linalg.norm(angle_differences, axis=1, ord=l)
            ))
            statistics.update(create_stats_ordered_dict(
                'L{} vel goal distance to target'.format(l),
                np.linalg.norm(vel_differences, axis=1, ord=l)
            ))

        for key, value in statistics.items():
            logger.record_tabular(key, value)


    @staticmethod
    def oc_reward(states, goal_states):
        return - torch.norm(states[:, :7] - goal_states[:, :7], p=2, dim=1)


class Reacher7DofCosSinFullGoalState(Reacher7DofFullGoalState):
    """
    The goal state is the full state: joint angles (in cos/sin parameterization)
    and velocities.
    """
    def _get_obs(self):
        angles = self.model.data.qpos.flat[:7]
        return np.concatenate([
            np.cos(angles),
            np.sin(angles),
            self.model.data.qvel.flat[:7],
        ])

    def sample_states(self, batch_size):
        full_states = super().sample_states(batch_size)
        angles = full_states[:, 7:]
        velocities = full_states[:, :7]
        return np.hstack((
            np.cos(angles),
            np.sin(angles),
            velocities
        ))

    @property
    def goal_dim(self):
        return 21

    def reset_model(self):
        saved_init_qpos = self.init_qpos.copy()
        qpos_tmp = self.init_qpos
        cos = self.multitask_goal[:7]
        sin = self.multitask_goal[7:14]
        angles = np.arctan2(sin, cos)
        qpos_tmp[:7] = angles
        qpos_tmp[7:] = self.multitask_goal[14:]
        qvel_tmp = np.zeros(self.model.nv)
        self.set_state(qpos_tmp, qvel_tmp)
        goal_xyz = self.get_body_com("tips_arm")

        # Now we actually set the goal position
        qpos = saved_init_qpos
        qpos[7:10] = goal_xyz
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv,
        )
        qvel[7:] = 0
        self.set_state(qpos, qvel)

        return self._get_obs()


"""
Reward functions for optimal control
"""
# DESIRED_JOINT_CONFIG = np.hstack((
#         np.random.uniform(
#             np.array([-2.28, -0.52, -1.4, -2.32, -1.5, -1.094, -1.5]),
#             np.array([1.71, 1.39, 1.7, 0, 1.5, 0, 1.5]),
#         ),
#         np.zeros(7),
#     ))

DESIRED_JOINT_CONFIG = np.array([  1.71470162e+00,  -5.23811120e-01,  -1.50068033e+00,
        -2.32198699e+00,  -6.56896692e-01,  -1.09489007e+00,
        -1.50093060e+00,  -4.39383377e-10,  -6.37245902e-10,
         8.24265517e-10,   4.09298618e-10,  -2.36288518e-06,
        -1.88265218e-10,   5.96086776e-10])
DESIRED_JOINT_CONFIG[7:] = 0
DESIRED_XYZ = np.array([-0.29606909, -0.18205661, -0.42400648])


def reach_a_joint_config_reward(states, *ignored):
    goal_pos = ptu.np_to_var(DESIRED_JOINT_CONFIG, requires_grad=False)
    return - torch.norm(states[:, :7] - goal_pos[:7], p=2, dim=1)
