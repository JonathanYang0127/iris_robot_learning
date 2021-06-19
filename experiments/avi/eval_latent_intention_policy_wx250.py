import argparse
import os
import torch
import numpy as np
import json
from PIL import Image

from train_latent_intention_policy import (
    TrajectoryConditionedPolicy,
    enable_gpus,
)
from rlkit.envs.wrappers.normalized_box_env import NormalizedBoxEnv
from widowx_envs.widowx.widowx_grasp_env import GraspWidowXEnv

import rlkit.torch.pytorch_util as ptu

from rlkit.misc.roboverse_utils import dump_video_basic

if __name__ == "__main__":
    num_trajs = 100
    full_image = True

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint-path", type=str, required=True)
    parser.add_argument("-v", "--video-save-dir", type=str, default="")
    parser.add_argument("-n", "--num-timesteps", type=int, default=10)
    parser.add_argument("--gpu", default='0', type=str)
    args = parser.parse_args()

    if not os.path.exists(args.video_save_dir):
        os.mkdir(args.video_save_dir)

    env = NormalizedBoxEnv(GraspWidowXEnv(
        {'transpose_image_to_chw': True,
         'wait_time': 0.2,
         'return_full_image': full_image}
    ))

    enable_gpus(args.gpu)
    ptu.set_gpu_mode(True)

    # Load json params
    checkpoint_dir = os.path.dirname(args.checkpoint_path)
    json_path = os.path.join(checkpoint_dir, 'variant.json')
    with open(json_path) as f:
        variant = json.load(f)
    action_dim = variant['action_dim']
    latent_dim = variant['latent_dim']
    rnn_hidden_size = variant['rnn_hidden_size']
    cnn_params = variant['cnn_params']
    batch_size = variant['batch_size']

    seq_cond_policy = TrajectoryConditionedPolicy(action_dim=action_dim,
                                                  latent_dim=latent_dim,
                                                  rnn_hidden_size=rnn_hidden_size,
                                                  trajectory_length=args.num_timesteps,
                                                  cnn_params=cnn_params,
                                                  batch_size=batch_size)
    seq_cond_policy.to(ptu.device)
    seq_cond_policy.load_state_dict(torch.load(args.checkpoint_path))
    seq_cond_policy.eval()

    paths = []

    for i in range(num_trajs):
        obs = env.reset()
        next_observations = []
        images = []

        z = np.random.randn(5,)

        if i > 0:
            input("Press Enter to continue...")
        print("Eval Traj {}".format(i))

        for _ in range(args.num_timesteps):
            obs = env._get_obs()
            next_observations.append(obs)
            action = seq_cond_policy.get_action(obs, z)
            obs, rew, done, info = env.step(action)

            if args.video_save_dir != "":
                if full_image:
                    image = obs['full_image']
                else:
                    image = np.uint8(np.reshape(obs['image'] * 255, (3, 64, 64)))
                    # image = np.transpose(image, (2, 1, 0)) (sideways image)
                    image = np.transpose(image, (1, 2, 0))
                images.append(Image.fromarray(image))

        paths.append(dict(next_observations=next_observations))

        # Save Video
        if args.video_save_dir != "":
            print("Saving Video")
            images[0].save('{}/eval_{}.gif'.format(args.video_save_dir, i),
                            format='GIF', append_images=images[1:],
                            save_all=True, duration=200, loop=0)
