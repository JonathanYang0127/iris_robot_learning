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

if __name__ == '__main__':
    num_trajs = 100
    full_image = True

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint-path", type=str, required=True)
    parser.add_argument("-v", "--video-save-dir", type=str, default="")
    parser.add_argument("-n", "--num-timesteps", type=int, default=10)
    args = parser.parse_args()

    if not os.path.exists(args.video_save_dir) and args.video_save_dir:
        os.mkdir(args.video_save_dir)

    ptu.set_gpu_mode(True)

    env = NormalizedBoxEnv(GraspWidowXEnv(
        {'transpose_image_to_chw': True,
         'wait_time': 0.2,
         'return_full_image': full_image}
    ))

    checkpoint_path = args.checkpoint_path
    _, ext = os.path.splitext(args.checkpoint_path)

    with open(args.checkpoint_path, 'rb') as handle:
        if ext == ".pt":
            params = torch.load(handle)
            eval_policy = params['evaluation/policy']
        elif ext == ".pkl":
            eval_policy = pickle.load(handle)

    eval_policy.eval()
    # eval_policy = params['exploration/policy']

    for i in range(num_trajs):
        obs = env.reset()
        # obs = env._get_obs()
        # obs_flat = ptu.from_numpy(np.append(obs['image'], obs['state']))

        images = []

        if i > 0:
            input("Press Enter to continue...")
        print("Eval Traj {}".format(i))

        obs = env._get_obs()
        obs_flat = ptu.from_numpy(np.append(obs['image'], obs['state']))

        for j in range(args.num_timesteps):
            action, info = eval_policy.get_action(obs_flat)
            # if j > 7:
            #     action[-1] = 0.0
            obs, rew, done, info = env.step(action)
            obs_flat = ptu.from_numpy(np.append(obs['image'], obs['state']))

            if args.video_save_dir:
                if full_image:
                    image = obs['full_image']
                else:
                    image = np.uint8(np.reshape(obs['image'] * 255, (3, 64, 64)))
                    # image = np.transpose(image, (2, 1, 0)) (sideways image)
                    image = np.transpose(image, (1, 2, 0))
                images.append(Image.fromarray(image))

        # Save Video
        if args.video_save_dir:
            print("Saving Video")
            images[0].save('{}/eval_{}.gif'.format(args.video_save_dir, i),
                            format='GIF', append_images=images[1:],
                            save_all=True, duration=200, loop=0)
