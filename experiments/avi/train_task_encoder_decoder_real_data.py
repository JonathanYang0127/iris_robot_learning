import os
import numpy as np
import argparse
import time
import pickle

from rlkit.data_management.multitask_replay_buffer import ObsDictMultiTaskReplayBuffer
from rlkit.misc.roboverse_utils import get_buffer_size_multitask
import roboverse
import rlkit.torch.pytorch_util as ptu

import torch.nn as nn
from torch import optim

from rlkit.launchers.config import LOCAL_LOG_DIR
from rlkit.core import logger
from rlkit.launchers.launcher_util import setup_logger
from rlkit.misc.wx250_utils import add_data_to_buffer_real_robot, DummyEnv

from rlkit.torch.task_encoders.encoder_decoder_nets import EncoderDecoderNet
from rlkit.torch.task_encoders.encoder_trainer import TaskEncoderTrainer


def add_reward_filtered_data_to_buffers_multitask(
        data, observation_keys,
        *args):
    for arg in args:
        assert len(arg) == 2
    for j in range(len(data)):
        path_len = len(data[j]['actions'])
        path = data[j]
        for i in range(path_len):
            for arg in args:
                if arg[1](path['rewards'][i]):
                    arg[0].add_sample(data[j]['env_infos'][0]['task_idx'],
                                      path['observations'][i], path['actions'][i], path['rewards'][i],
                                      path['terminals'][i], path['next_observations'][i]
                                      )


def enable_gpus(gpu_str):
    if gpu_str != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
    return


BUFFER = ('/home/avi/Downloads/lego_pickle_train.pkl')
VALIDATION_BUFFER = ('/home/avi/Downloads/lego_pickle_val.pkl')
ENV = 'Widow250PickPlaceMetaTestMultiObjectMultiContainer-v0'


def main(args):
    variant = dict(
        buffer=args.buffer,
        val_buffer=args.val_buffer,
        beta_target=args.beta_target,
        beta_anneal_steps=args.beta_anneal_steps,
        latent_dim=args.latent_dim,
        encoder_resnet=args.encoder_resnet,
        decoder_resnet=args.decoder_resnet,
        total_steps=int(5e5),
        batch_size=args.batch_size,
        num_tasks=args.num_tasks,
    )

    enable_gpus(args.gpu)
    ptu.set_gpu_mode(True)

    with open(variant['buffer'], 'rb') as fl:
        data = np.load(fl, allow_pickle=True)

    num_transitions = get_buffer_size_multitask(data)
    max_replay_buffer_size = num_transitions + 10
    image_size = 64
    expl_env = DummyEnv(image_size=image_size, use_wrist=True)
    train_task_indices = list(range(variant['num_tasks']))
    observation_keys = ['image',]

    replay_buffer_positive = ObsDictMultiTaskReplayBuffer(
        int(max_replay_buffer_size/2),
        expl_env,
        train_task_indices,
        use_next_obs_in_context=False,
        sparse_rewards=False,
        observation_keys=observation_keys
    )

    replay_buffer_full = ObsDictMultiTaskReplayBuffer(
        max_replay_buffer_size,
        expl_env,
        train_task_indices,
        use_next_obs_in_context=False,
        sparse_rewards=False,
        observation_keys=observation_keys
    )
    # train_task_indices = list(range(32))

    add_reward_filtered_data_to_buffers_multitask(data, observation_keys,
                                                  (replay_buffer_positive, lambda r: r > 0),
                                                  (replay_buffer_full, lambda r: True))

    with open(variant['val_buffer'], 'rb') as fl:
        data = np.load(fl, allow_pickle=True)

    num_transitions = get_buffer_size_multitask(data)
    max_replay_buffer_size = num_transitions + 10

    replay_buffer_positive_val = ObsDictMultiTaskReplayBuffer(
        int(max_replay_buffer_size/2),
        expl_env,
        train_task_indices,
        use_next_obs_in_context=False,
        sparse_rewards=False,
        observation_keys=observation_keys
    )
    replay_buffer_full_val = ObsDictMultiTaskReplayBuffer(
        max_replay_buffer_size,
        expl_env,
        train_task_indices,
        use_next_obs_in_context=False,
        sparse_rewards=False,
        observation_keys=observation_keys
    )
    add_reward_filtered_data_to_buffers_multitask(data, observation_keys,
                                                  (replay_buffer_positive_val, lambda r: r > 0),
                                                  (replay_buffer_full_val, lambda r: True))

    latent_dim = variant['latent_dim']
    net = EncoderDecoderNet(image_size, latent_dim, encoder_resent=args.encoder_resnet)
    net.to(ptu.device)
    exp_prefix = '{}-task-encoder-decoder'.format(time.strftime("%y-%m-%d"))
    setup_logger(logger, exp_prefix, LOCAL_LOG_DIR, variant=variant,
                 snapshot_mode='gap_and_last', snapshot_gap=100, )


    batch_size = variant['batch_size']

    optimizer = optim.Adam(net.parameters(), lr=3e-4)
    # log_alpha = ptu.zeros(1, requires_grad=True)
    # alpha_optimizer = optim.Adam([log_alpha], lr=0.01)

    print_freq = 100
    total_steps = variant['total_steps']
    half_beta_target_steps = min(total_steps // 2, args.beta_anneal_steps)
    beta_target = args.beta_target
    criterion = nn.CrossEntropyLoss()
    tasks_to_sample = list(range(variant['num_tasks']))

    trainer = TaskEncoderTrainer(net, optimizer, criterion, print_freq,
                                 beta_target, half_beta_target_steps, args.anneal)

    trainer.train(replay_buffer_full, replay_buffer_positive, replay_buffer_full_val,
                  replay_buffer_positive_val, total_steps, batch_size, tasks_to_sample, logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer", type=str, default=BUFFER)
    parser.add_argument("--val-buffer", type=str, default=VALIDATION_BUFFER)
    parser.add_argument("--num-tasks", type=int, default=2)
    parser.add_argument("--anneal", type=str, default='linear',
                        choices=('sigmoid', 'linear', 'none'))
    parser.add_argument("--latent-dim", default=2, type=int)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gpu", default='0', type=str)
    parser.add_argument("--beta-target", type=float, default=0.001)
    parser.add_argument("--beta-anneal-steps", type=int, default=10000)
    parser.add_argument("--encoder-resnet", default=False, action='store_true')
    parser.add_argument("--decoder-resnet", default=False, action='store_true')
    parser.add_argument("--use-alpha", default=False, action='store_true')
    args = parser.parse_args()
    main(args)