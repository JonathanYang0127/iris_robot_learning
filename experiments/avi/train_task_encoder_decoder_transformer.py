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

from rlkit.torch.task_encoders.encoder_decoder_nets import TransformerEncoderDecoderNet
from rlkit.torch.task_encoders.transformer_encoder_trainer import TransformerTaskEncoderTrainer


def add_reward_filtered_trajectories_to_buffers_multitask(
        data, observation_keys,
        *args):
    for arg in args:
        assert len(arg) == 2
    for j in range(len(data)):
        path_len = len(data[j]['actions'])
        path = data[j]
        task_idx = data[j]['env_infos'][0]['task_idx']
        
        for arg in args:
            if arg[1](path['rewards']):
                for i in range(path_len):
                    path['observations'][i]['image'] = process_image(path['observations'][i]['image'])
                    path['next_observations'][i]['image'] = process_image(path['next_observations'][i]['image'])
                    arg[0].add_sample(data[j]['env_infos'][0]['task_idx'],
                                      path['observations'][i], path['actions'][i], path['rewards'][i],
                                      path['terminals'][i], path['next_observations'][i]
                                      )

def add_reward_filtered_data_to_buffers_multitask(
        data, observation_keys,
        *args):
    for arg in args:
        assert len(arg) == 2
    for j in range(len(data)):
        path_len = len(data[j]['actions'])
        path = data[j]
        task_idx = path['env_infos'][0]['task_idx']
        for i in range(path_len):
            for arg in args:
                path['observations'][i]['image'] = process_image(path['observations'][i]['image'])
                path['next_observations'][i]['image'] = process_image(path['next_observations'][i]['image'])
                if arg[1](path['rewards'][i]):
                    arg[0].add_sample(data[j]['env_infos'][0]['task_idx'],
                                      path['observations'][i], path['actions'][i], path['rewards'][i],
                                      path['terminals'][i], path['next_observations'][i]
                                      )

def process_image(image):
    if len(image.shape) == 3:
        image = np.transpose(image, [2, 0, 1])
        image = (image.flatten())

    if np.mean(image) > 5:
        image = image / 255.0
    return image


def enable_gpus(gpu_str):
    if gpu_str != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
    return


BUFFER = ('/nfs/kun1/users/jonathan/robotnetv2_data/july_15_kulbot_train.pkl')
VALIDATION_BUFFER = ('/nfs/kun1/users/jonathan/robotnetv2_data/july_15_kulbot_val.pkl')


def main(args):
    variant = dict(
        env=args.env,
        buffer=args.buffer,
        val_buffer=args.val_buffer,
        beta_target=args.beta_target,
        beta_anneal_steps=args.beta_anneal_steps,
        latent_dim=args.latent_dim,
        decoder_resnet=args.decoder_resnet,
        total_steps=int(5e5),
        meta_batch_size=args.meta_batch_size,
        batch_size=args.batch_size,
        num_tasks=args.num_tasks,
        image_augmentation=args.use_image_aug,
        encoder_keys=['observations', 'actions'],
        path_len=args.path_len
    )

    enable_gpus(args.gpu)
    ptu.set_gpu_mode(True)

    with open(variant['buffer'], 'rb') as fl:
        data = np.load(fl, allow_pickle=True)

    assert len(data[0]['actions']) == variant['path_len']
    num_transitions = get_buffer_size_multitask(data)
    max_replay_buffer_size = num_transitions + 10
    image_size = 48
    expl_env = roboverse.make(variant['env'], transpose_image=True, num_tasks=variant['num_tasks'])
    train_task_indices = list(range(variant['num_tasks']))
    print(train_task_indices)
    observation_keys = ['image',]

    buffer_kwargs = {
        'use_next_obs_in_context': False,
        'sparse_rewards': False,
        'observation_keys': observation_keys
    }
    traj_buffer_positive = ObsDictMultiTaskReplayBuffer(
        int(max_replay_buffer_size),
        expl_env,
        train_task_indices,
        path_len=variant['path_len'],
        observation_keys=['image', 'state'],
        use_next_obs_in_context=False,
        sparse_rewards=False
    )
    replay_buffer_positive = ObsDictMultiTaskReplayBuffer(
        int(max_replay_buffer_size),
        expl_env,
        train_task_indices,
        path_len=variant['path_len'],
        **buffer_kwargs
    )
    replay_buffer_full = ObsDictMultiTaskReplayBuffer(
        max_replay_buffer_size,
        expl_env,
        train_task_indices,
        path_len=variant['path_len'],
        **buffer_kwargs
    )
    import pickle

    # train_task_indices = list(range(32))

    add_reward_filtered_trajectories_to_buffers_multitask(data, observation_keys,
                                                  (traj_buffer_positive, lambda r: np.sum(r) > 0),
                                                  (replay_buffer_full, lambda r: True))
    add_reward_filtered_data_to_buffers_multitask(data, observation_keys,
                                                  (replay_buffer_positive, lambda r: r > 0))
    with open('/home/jonathan/traj_sim.pkl', 'wb') as f:
        pickle.dump(traj_buffer_positive, f)

    with open(variant['val_buffer'], 'rb') as fl:
        data = np.load(fl, allow_pickle=True)

    assert len(data[0]['actions']) == variant['path_len'], "Data len {} doesn't match {}".format(len(data[0]['actions']), variant['path_len'])
    num_transitions = get_buffer_size_multitask(data)
    max_replay_buffer_size = num_transitions + 10

    traj_buffer_positive_val = ObsDictMultiTaskReplayBuffer(
        int(max_replay_buffer_size),
        expl_env,
        train_task_indices,
        path_len=variant['path_len'],
        observation_keys=['image', 'state'],
        use_next_obs_in_context=False,
        sparse_rewards=False
    )
    replay_buffer_positive_val = ObsDictMultiTaskReplayBuffer(
        int(max_replay_buffer_size),
        expl_env,
        train_task_indices,
        path_len=variant['path_len'],
        **buffer_kwargs
    )
    replay_buffer_full_val = ObsDictMultiTaskReplayBuffer(
        max_replay_buffer_size,
        expl_env,
        train_task_indices,
        path_len=variant['path_len'],
        **buffer_kwargs
    )
    add_reward_filtered_trajectories_to_buffers_multitask(data, observation_keys,
                                                  (traj_buffer_positive_val, lambda r: np.sum(r) > 0),
                                                  (replay_buffer_full_val, lambda r: True))
    add_reward_filtered_data_to_buffers_multitask(data, observation_keys,
                                                  (replay_buffer_positive, lambda r: r > 0))

    latent_dim = variant['latent_dim']
    net = TransformerEncoderDecoderNet(image_size, latent_dim, variant['num_tasks'], variant['path_len'],
                            image_augmentation=args.use_image_aug,
                            encoder_keys=variant['encoder_keys'])
    net.to(ptu.device)
    exp_prefix = '{}-task-encoder-decoder-transformer'.format(time.strftime("%y-%m-%d"))
    save_freq = 100
    setup_logger(logger, exp_prefix, LOCAL_LOG_DIR, variant=variant,
                 snapshot_mode='gap_and_last', snapshot_gap=save_freq, )


    meta_batch_size = variant['meta_batch_size']
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

    trainer = TransformerTaskEncoderTrainer(net, optimizer, criterion, print_freq, save_freq,
                                 beta_target, half_beta_target_steps, args.anneal, 
                                 encoder_keys=variant['encoder_keys'])

    tasks_to_sample = np.arange(variant['num_tasks'])
    trainer.train(replay_buffer_full, replay_buffer_positive, traj_buffer_positive, replay_buffer_full_val,
                  replay_buffer_positive_val, traj_buffer_positive_val, total_steps, meta_batch_size, batch_size,
                  tasks_to_sample, logger)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--buffer", type=str, default=BUFFER)
    parser.add_argument("--val-buffer", type=str, default=VALIDATION_BUFFER)
    parser.add_argument("--num-tasks", type=int, default=2)
    parser.add_argument("--anneal", type=str, default='linear',
                        choices=('sigmoid', 'linear', 'none'))
    parser.add_argument("--path-len", default=30, type=int)
    parser.add_argument("--latent-dim", default=2, type=int)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--meta-batch-size", type=int, default=8)
    parser.add_argument("--gpu", default='0', type=str)
    parser.add_argument("--beta-target", type=float, default=0.001)
    parser.add_argument("--beta-anneal-steps", type=int, default=10000)
    parser.add_argument("--decoder-resnet", default=False, action='store_true')
    parser.add_argument("--use-image-aug", default=False, action='store_true')
    parser.add_argument("--use-alpha", default=False, action='store_true')
    args = parser.parse_args()
    main(args)
