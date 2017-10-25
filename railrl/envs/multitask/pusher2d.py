import abc

import numpy as np
import torch

from railrl.envs.mujoco.pusher2d import Pusher2DEnv
from railrl.envs.multitask.multitask_env import MultitaskEnv


class MultitaskPusher2DEnv(Pusher2DEnv, MultitaskEnv, metaclass=abc.ABCMeta):
    def __init__(self, goal=(0, -1)):
        self.init_serialization(locals())
        super().__init__(goal=goal)
        MultitaskEnv.__init__(self)

    def sample_actions(self, batch_size):
        return np.random.uniform(self.action_space.low, self.action_space.high)

    def sample_states(self, batch_size):
        raise NotImplementedError()

    def log_diagnostics(self, paths):
        super().log_diagnostics(paths)
        MultitaskEnv.log_diagnostics(self, paths)


class FullStatePusher2DEnv(MultitaskPusher2DEnv):
    def sample_goal_states(self, batch_size):
        # Joint angle and xy position won't be consistent, but oh well!
        return np.random.uniform(
            np.array([-2.5, -2.3213, -2.3213, -1, -1, -1, -1, -1, -1, -1]),
            np.array([2.5, 2.3, 2.3, 1, 1, 1, 0, 1, 0, 1]),
            (batch_size, self.goal_dim)
        )

    @property
    def goal_dim(self):
        return 10

    def convert_obs_to_goal_states(self, obs):
        return obs

    def set_goal(self, goal):
        super().set_goal(goal)
        self._target_cylinder_position = goal[-2:]
        self._target_hand_position = goal[-4:-2]

        qpos = self.model.data.qpos.flat.copy()
        qvel = self.model.data.qvel.flat.copy()
        qpos[-4:-2] = self._target_cylinder_position
        qpos[-2:] = self._target_hand_position
        self.set_state(qpos, qvel)

    @staticmethod
    def move_hand_to_cylinder_oc_reward_on_goals(
            predicted_states, ignored, current_states
    ):
        return -torch.norm(
            predicted_states[:, 6:8]
            - current_states[:, 8:10]
        )

    @staticmethod
    def move_hand_to_target_position_oc_reward_on_goals(
            predicted_states, goal_states, current_states
    ):
        return -torch.norm(
            predicted_states[:, 6:8]
            - goal_states[:, 6:8]
        )


# Stupid pickle
def FullStatePusher2DEnv_move_hand_to_target_position_oc_reward_on_goals(
        predicted_states, goal_states, current_states
):
    return -torch.norm(
        predicted_states[:, 6:8]
        - goal_states[:, 6:8]
    )

def FullStatePusher2DEnv_move_hand_to_cylinder_oc_reward_on_goals(
        predicted_states, goal_states, current_states
):
    return -torch.norm(
        predicted_states[:, 6:8]
        - current_states[:, 6:8]
    )


class HandCylinderXYPusher2DEnv(MultitaskPusher2DEnv):
    def sample_goal_states(self, batch_size):
        return np.random.uniform(
            np.array([-1, -1, -1., -1]),
            np.array([0, 1, 0, 1]),
            (batch_size, self.goal_dim)
        )

    @property
    def goal_dim(self):
        return 4

    def convert_obs_to_goal_states(self, obs):
        return obs[:, -4:]

    def set_goal(self, goal):
        super().set_goal(goal)
        self._target_cylinder_position = goal[-2:]
        self._target_hand_position = goal[-4:-2]

        qpos = self.model.data.qpos.flat.copy()
        qvel = self.model.data.qvel.flat.copy()
        qpos[-4:-2] = self._target_cylinder_position
        qpos[-2:] = self._target_hand_position
        self.set_state(qpos, qvel)


class HandXYPusher2DEnv(MultitaskPusher2DEnv):
    """
    Only care about the hand position! This is really just for debugging.
    """
    def sample_goal_states(self, batch_size):
        return np.random.uniform(
            np.array([-1, -1]),
            np.array([0, 1]),
            (batch_size, self.goal_dim)
        )

    @property
    def goal_dim(self):
        return 2

    def convert_obs_to_goal_states(self, obs):
        return obs[:, -4:-2]

    def set_goal(self, goal):
        super().set_goal(goal)
        self._target_hand_position = goal

        qpos = self.model.data.qpos.flat.copy()
        qvel = self.model.data.qvel.flat.copy()
        qpos[-2:] = self._target_hand_position
        self.set_state(qpos, qvel)

    @staticmethod
    def oc_reward(states, goals):
        """
        Reminder:

        ```
        def _get_obs(self):
            return np.concatenate([
                self.model.data.qpos.flat[:3],
                self.model.data.qvel.flat[:3],
                self.get_body_com("distal_4")[:2],
                self.get_body_com("object")[:2],
            ])
        ```

        :param states:
        :param goals:
        :return:
        """
        return HandXYPusher2DEnv.oc_reward_on_goals(states[:, 6:8], goals)

    @staticmethod
    def oc_reward_on_goals(goals_predicted, goals):
        return - torch.norm(goals_predicted - goals)


class FixedHandXYPusher2DEnv(HandXYPusher2DEnv):
    def sample_goal_state_for_rollout(self):
        return np.array([-1, 0])


class CylinderXYPusher2DEnv(MultitaskPusher2DEnv):
    def sample_goal_states(self, batch_size):
        return np.random.uniform(
            np.array([-1, -1]),
            np.array([0, 1]),
            (batch_size, self.goal_dim)
        )

    @property
    def goal_dim(self):
        return 2

    def convert_obs_to_goal_states(self, obs):
        return obs[:, -2:]

    def set_goal(self, goal):
        super().set_goal(goal)
        self._target_cylinder_position = goal

        qpos = self.model.data.qpos.flat.copy()
        qvel = self.model.data.qvel.flat.copy()
        qpos[-4:-2] = self._target_cylinder_position
        self.set_state(qpos, qvel)

"""
Optimal control rewards
"""
