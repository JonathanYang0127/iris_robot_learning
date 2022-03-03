from widowx_envs.widowx.widowx_grasp_env import GraspWidowXEnv
# from widowx_envs.policies.scripted_grasp import GraspPolicy
from rlkit.envs.wrappers.normalized_box_env import NormalizedBoxEnv

import os
import os.path as osp
import glob
from widowx_envs.widowx.widowx_env import BridgeDataRailRLPrivateWidowX

import argparse
import os
import pickle
import torch
import numpy as np
from PIL import Image
import time

import rlkit.torch.pytorch_util as ptu

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from rlkit.torch.sac.policies import MakeDeterministic

right_front = [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam_initial_testconfig/rss_benchmark/toykitchen1/put_broccoli_in_pot/2022-01-22_11-06-56/raw/traj_group0/traj0', 150]
middle = [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam_initial_testconfig/rss_benchmark/toykitchen1/put_broccoli_in_pot/2022-01-22_11-06-56/raw/traj_group0/traj1', 290]
left_front = [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam_initial_testconfig/rss_benchmark/toykitchen1/put_broccoli_in_pot/2022-01-22_11-06-56/raw/traj_group0/traj2', 200]
left_rear = [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam_initial_testconfig/rss_benchmark/toykitchen1/put_broccoli_in_pot/2022-01-22_17-54-21/raw/traj_group0/traj0', 290]
right_rear = [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam_initial_testconfig/rss_benchmark/toykitchen1/put_broccoli_in_pot/2022-01-22_17-54-21/raw/traj_group0/traj1', 290]

class max_q_policy:
    def __init__(self, qf1, policy, use_robot_state=True, num_repeat=100):
        self.qf1 = qf1
        self.policy = policy
        self.use_robot_state = use_robot_state
        self.num_repeat = num_repeat

    def get_action(self, obs):
        """
        Used when sampling actions from the policy and doing max Q-learning
        """
        with torch.no_grad():
            obs = obs.view(1, -1).repeat(self.num_repeat, 1)
            action = self.policy(obs).rsample()
            q1 = self.qf1(obs, action)
            print(q1)
            ind = q1.max(0)[1]
        return ptu.get_numpy(action[ind]).flatten(), None


    def eval(self):
        self.qf1.eval()
        self.policy.eval()


if __name__ == '__main__':
    num_trajs = 100
    full_image = True

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint-path", type=str, required=True)
    parser.add_argument("-v", "--video-save-dir", type=str, default="")
    parser.add_argument("-n", "--num-timesteps", type=int, default=15)
    parser.add_argument("--q-value-eval", default=False, action='store_true')
    parser.add_argument("--num-tasks", type=int, default=0)
    parser.add_argument("--task-embedding", default=False, action="store_true")
    parser.add_argument("--task-encoder", default=None)
    parser.add_argument("--sample-trajectory", type=str, default=None)
    parser.add_argument("--use-checkpoint-encoder", action='store_true', default=False)
    args = parser.parse_args()

    assert args.num_tasks != 0 or args.task_embedding
    if not os.path.exists(args.video_save_dir) and args.video_save_dir:
        os.mkdir(args.video_save_dir)

    ptu.set_gpu_mode(True)

    env_params = {
        'fix_zangle': 0.1,
        'move_duration': 0.2,
        'adaptive_wait': True,
        'move_to_rand_start_freq': 1,
        'override_workspace_boundaries': [[0.17, -0.08, 0.06, -1.57, 0], [0.35, 0.08, 0.1,  1.57, 0]],  # [[0.2, -0.04, 0.03, -1.57, 0], [0.31, 0.04, 0.1,  1.57, 0]],
        'action_clipping': None,
        'catch_environment_except':True,
        'start_transform': right_rear,
    }

    env = BridgeDataRailRLPrivateWidowX(env_params)

    checkpoint_path = args.checkpoint_path
    _, ext = os.path.splitext(args.checkpoint_path)

    with open(args.checkpoint_path, 'rb') as handle:
        if ext == ".pt":
            params = torch.load(handle)
            if args.q_value_eval:
                eval_policy = max_q_policy(params['trainer/qf1'], params['trainer/policy'])
            else:
                eval_policy = params['evaluation/policy']
        elif ext == ".pkl":
            eval_policy = pickle.load(handle)

    eval_policy.eval()
    eval_policy = MakeDeterministic(eval_policy)

    env.start()

    for i in range(num_trajs):
        obs = env.reset()

        images = []

        valid_task_idx = False
        while not valid_task_idx:
            task_idx = "None"
            while not task_idx.isnumeric():
                task_idx = input("Enter task idx to continue...")
            task_idx = int(task_idx)
            valid_task_idx = task_idx in list(range(args.num_tasks))
        task = np.array([0] * args.num_tasks)
        task[task_idx] = 1
        print("Eval Traj {}".format(i))

        obs = env._get_obs()

        # plt.imshow(obs['image'][0].reshape(3, 128, 128).transpose(1, 2, 0))
        # plt.savefig("test.png")

        obs_flat = ptu.from_numpy(np.concatenate([obs['image'][0], task]))

        for j in range(args.num_timesteps):
            tstamp_return_obs = time.time()

            action, info = eval_policy.get_action(obs_flat)
            tstamp_return_obs += 0.2
            obs, rew, done, info = env.step(action, get_obs_tstamp=tstamp_return_obs, blocking=False)
            obs_flat = ptu.from_numpy(np.concatenate([obs['image'][0], task]))

            if args.video_save_dir:
                if full_image:
                    image = obs['full_image']
                else:
                    image = np.uint8(np.reshape(obs['image'] * 255, (3, 64, 64)))
                    image = np.transpose(image, (1, 2, 0))
                images.append(Image.fromarray(image))

        # Save Video
        if args.video_save_dir:
            print("Saving Video")
            images[0].save('{}/eval_{}.gif'.format(args.video_save_dir, i),
                            format='GIF', append_images=images[1:],
                            save_all=True, duration=200, loop=0)
