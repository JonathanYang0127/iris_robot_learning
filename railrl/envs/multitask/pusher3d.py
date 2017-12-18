from railrl.envs.mujoco.mujoco_env import MujocoEnv
import numpy as np

from railrl.envs.multitask.multitask_env import MultitaskEnv


class MultitaskPusher3DEnv(MujocoEnv, MultitaskEnv):
    GOAL_ZERO_POS = [-0.35, -0.35, -0.3230]  # from xml
    OBJ_ZERO_POS = [0.35, -0.35, -0.275]  # from xml
    goal_low = [-0.4, -0.4]
    goal_high = [0.4, 0.0]

    def __init__(self):
        self.init_serialization(locals())
        MultitaskEnv.__init__(self)
        MujocoEnv.__init__(
            self,
            "pusher_3d.xml",
            5,
            automatically_set_obs_and_action_space=True,
        )

    def _step(self, a):
        obj_to_arm = self.get_body_com("object") - self.get_body_com("tips_arm")
        obj_to_goal = self.get_body_com("object") - self.get_body_com("goal")
        # Only care about x and y axis.
        obj_to_arm = obj_to_arm[:2]
        obj_to_goal = obj_to_goal[:2]
        obj_to_arm_dist = np.linalg.norm(obj_to_arm)
        obj_to_goal_dist = np.linalg.norm(obj_to_goal)
        control_dist = np.linalg.norm(a)

        forward_reward_vec = [obj_to_goal_dist, obj_to_arm_dist, control_dist]
        reward_coefs = (0.5, 0.375, 0.125)
        reward = -sum(
            [coef * r for (coef, r) in zip(reward_coefs, forward_reward_vec)]
        )

        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False

        return ob, reward, done, dict(
            obj_to_arm_dist=obj_to_arm_dist,
            obj_to_goal_dist=obj_to_goal_dist,
            control_dist=control_dist,
        )

    def _get_obs(self):
        obs = np.concatenate([
            self.model.data.qpos.flat[:3],
            self.model.data.qvel.flat[:3],
            self.get_body_com("tips_arm")[:2],
            self.get_body_com("object")[:2],
        ])
        return obs

    def reset_model(self):
        qpos = self.init_qpos
        qpos[:] = 0
        qpos[-4:-2] += self.np_random.uniform(-0.05, 0.05, 2)
        qpos[-2:] = self.multitask_goal
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv)
        qvel[-4:] = 0

        self.set_state(qpos, qvel)

        return self._get_obs()

    def sample_goals(self, batch_size):
        return np.random.uniform(
            self.goal_low,
            self.goal_high,
            (batch_size, self.goal_dim)
        )

    def convert_obs_to_goals(self, obs):
        return obs[:, 8:10]

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 4.0

    @property
    def goal_dim(self) -> int:
        return 2
