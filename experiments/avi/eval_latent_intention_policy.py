
import argparse
import os
import torch
import numpy as np
import json

from train_latent_intention_policy import TrajectoryConditionedPolicy, \
    enable_gpus
import rlkit.torch.pytorch_util as ptu

import roboverse

CHECKPOINT = '/media/avi/data/Work/doodad_output/21-06-16-latent_intention_model_2021_06_16_16_10_37_id000--s0/itr_490.pt'

from rlkit.misc.roboverse_utils import dump_video_basic

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT)
    parser.add_argument("--gpu", default='0', type=str)
    args = parser.parse_args()

    enable_gpus(args.gpu)
    ptu.set_gpu_mode(True)
    checkpoint_dir = os.path.dirname(args.checkpoint)
    json_path = os.path.join(checkpoint_dir, 'variant.json')
    with open(json_path) as f:
        variant = json.load(f)

    action_dim = variant['action_dim']
    latent_dim = variant['latent_dim']
    rnn_hidden_size = variant['rnn_hidden_size']
    # state_observation_dim = variant[]
    trajectory_length = 30
    cnn_params = variant['cnn_params']

    batch_size = variant['batch_size']
    seq_cond_policy = TrajectoryConditionedPolicy(action_dim=action_dim,
                                                  latent_dim=latent_dim,
                                                  rnn_hidden_size=rnn_hidden_size,
                                                  trajectory_length=trajectory_length,
                                                  cnn_params=cnn_params,
                                                  batch_size=batch_size)
    seq_cond_policy.to(ptu.device)
    seq_cond_policy.load_state_dict(torch.load(args.checkpoint))
    seq_cond_policy.eval()

    variant = dict(env='Widow250PickPlaceMedium-v0')

    env = roboverse.make(variant['env'], transpose_image=True)
    # env = roboverse.make(variant['env'], transpose_image=True, gui=True)
    paths = []
    env.reset()

    for _ in range(10):
        z = np.random.randn(5,)
        next_observations = []
        env.reset_robot_only()
        for _ in range(35):
            obs = env.get_observation()
            next_observations.append(obs)
            action = seq_cond_policy.get_action(obs, z)
            obs, rew, done, info = env.step(action)

        paths.append(dict(next_observations=next_observations))

    video_dir = '/media/avi/data/Work/doodad_output/test_seq_videos'
    dump_video_basic(video_dir, paths)
