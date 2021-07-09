# from widowx_envs.widowx.widowx_grasp_env import GraspWidowXEnv
# from widowx_envs.policies.scripted_grasp import GraspPolicy
# from rlkit.envs.wrappers.normalized_box_env import NormalizedBoxEnv

import roboverse
import argparse
import os
import pickle
import torch
import numpy as np
from PIL import Image
from awac_image_multitask import EmbeddingWrapper
import rlkit.torch.pytorch_util as ptu
import json

if __name__ == '__main__':
    num_trajs = 12
    full_image = False

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint-path", type=str, required=True)
    parser.add_argument("-v", "--video-save-dir", type=str, default="")
    # parser.add_argument("-n", "--num-timesteps", type=int, default=10)
    # parser.add_argument("-e", "--env", type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.video_save_dir) and args.video_save_dir:
        os.mkdir(args.video_save_dir)

    # Load variants
    variants_path = os.path.dirname(args.checkpoint_path)
    variants_path = os.path.join(variants_path, "variant.json")
    with open(variants_path) as variants_json:
        variants = json.load(variants_json)

    ptu.set_gpu_mode(True)

    env = roboverse.make(variants['env'], transpose_image=True, num_tasks=variants['num_tasks'])

    _, ext = os.path.splitext(args.checkpoint_path)

    with open(args.checkpoint_path, 'rb') as handle:
        if ext == ".pt":
            params = torch.load(handle)
            eval_policy = params['evaluation/policy']
        elif ext == ".pkl":
            eval_policy = pickle.load(handle)

    eval_policy.eval()
    # eval_policy = params['exploration/policy']
    task_embs = [
        np.array([-0.05158421, 1.3213843]),
        np.array([-0.00895223, -0.7244007 ]),
        np.array([0.00860909, 2.5347257 ]),
        np.array([-0.00430241, -0.24293025]),
        np.array([-0.0124189 ,  0.20121233]),
        np.array([-0.0197675,  0.7054519]),
    ]

    for i in range(num_trajs):
        obs = env.reset()
        # obs = env._get_obs()
        # obs_flat = ptu.from_numpy(np.append(obs['image'], obs['state']))

        images = []

        print("Eval Traj {}".format(i))

        obs = env.get_observation()
        task_emb = task_embs[i % len(task_embs)]
        added_fc_input = np.concatenate((obs['state'], task_emb))
        obs_flat = ptu.from_numpy(np.append(obs['image'], added_fc_input))
        print("obs_flat.shape", obs_flat.shape)

        for j in range(variants['max_path_length']):
            action, info = eval_policy.get_action(obs_flat)
            # if j > 7:
            #     action[-1] = 0.0
            obs, rew, done, info = env.step(action)
            added_fc_input = np.concatenate((obs['state'], task_emb))
            obs_flat = ptu.from_numpy(np.append(obs['image'], added_fc_input))

            if args.video_save_dir:
                if full_image:
                    image = obs['full_image']
                else:
                    image = np.uint8(np.reshape(obs['image'] * 255, (3, 48, 48)))
                    # image = np.transpose(image, (2, 1, 0)) (sideways image)
                    image = np.transpose(image, (1, 2, 0))
                images.append(Image.fromarray(image))

        # Save Video
        if args.video_save_dir:
            print("Saving Video")
            images[0].save('{}/eval_{}.gif'.format(args.video_save_dir, i),
                            format='GIF', append_images=images[1:],
                            save_all=True, duration=200, loop=0)
