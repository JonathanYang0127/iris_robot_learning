from widowx_envs.widowx.widowx_grasp_env import GraspWidowXEnv
# from widowx_envs.policies.scripted_grasp import GraspPolicy
from rlkit.envs.wrappers.normalized_box_env import NormalizedBoxEnv

import argparse
import os
import pickle
import torch
import numpy as np
from PIL import Image

import rlkit.torch.pytorch_util as ptu

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
    args = parser.parse_args()

    assert args.num_tasks != 0 or args.task_embedding
    if not os.path.exists(args.video_save_dir) and args.video_save_dir:
        os.mkdir(args.video_save_dir)

    ptu.set_gpu_mode(True)

    env = NormalizedBoxEnv(GraspWidowXEnv(
        {'transpose_image_to_chw': True,
         'wait_time': 0.2,
         'return_full_image': full_image,
         'action_mode': '3trans1rot'}
    ))

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

    if args.task_encoder:
        from rlkit.torch.task_encoders.encoder_decoder_nets import EncoderDecoderNet
        net = EncoderDecoderNet(64, 2, encoder_resnet=False)
        net.load_state_dict(torch.load(args.task_encoder))
        net.to(ptu.device)

    for i in range(num_trajs):
        obs = env.reset()

        images = []

        if not args.task_embedding:
            valid_task_idx = False
            while not valid_task_idx:
                task_idx = "None"
                while not task_idx.isnumeric():
                    task_idx = input("Enter task idx to continue...")
                task_idx = int(task_idx)
                valid_task_idx = task_idx in list(range(args.num_tasks))
            task = np.array([0] * args.num_tasks)
            task[task_idx] = 1
        else:
            if args.task_encoder:
                input("Press enter to take image")
                obs = ptu.from_numpy(env.reset()['image'].reshape(1, 1, -1))
                task = ptu.get_numpy(net.encoder_net(obs)[1])
                task = task.reshape(-1)
                print("task: ", task)
                input("Press enter to continue")
            else:
                task = "None"
                while not isinstance(eval(task), list):
                    task = input("Enter task embedding to continue...")
                task = np.array(eval(task))
        print("Eval Traj {}".format(i))

        obs = env._get_obs()
        obs_flat = ptu.from_numpy(np.concatenate([obs['image'], obs['state'], task]))

        for j in range(args.num_timesteps):
            action, info = eval_policy.get_action(obs_flat)
            obs, rew, done, info = env.step(action)
            obs_flat = ptu.from_numpy(np.concatenate([obs['image'], obs['state'], task]))

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
