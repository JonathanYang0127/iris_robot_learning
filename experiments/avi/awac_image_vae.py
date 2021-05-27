import argparse
import time
import os.path as osp
import os

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from rlkit.torch.sac.awac_trainer import AWACTrainer
from rlkit.torch.sac.policies import GaussianVQVAEPolicy, MakeDeterministic
from rlkit.torch.networks.vqvae import VQVAEWrapper, ConcatVQVAEWrapper

from rlkit.data_management.obs_dict_replay_buffer import ObsDictReplayBufferVQVAE
from rlkit.samplers.data_collector import MdpPathCollector, ObsDictPathCollector
from rlkit.launchers.launcher_util import setup_logger
from rlkit.core import logger
from rlkit.torch.networks import Clamp
from rlkit.pythonplusplus import identity
from rlkit.misc.roboverse_utils import add_data_to_buffer, VideoSaveFunctionBullet


import roboverse
import numpy as np
import torch
import sys

VQVAE_DIR = '/home/jonathanyang0127/vqvae'
from rlkit.launchers.config import LOCAL_LOG_DIR


BUFFER = '/home/jonathanyang0127/minibullet/data/may18_Widow250OneObjectGraspTrain-v0_20K_save_all_noise_0.1_2021-05-18T21-59-01/may18_Widow250OneObjectGraspTrain-v0_20K_save_all_noise_0.1_2021-05-18T21-59-01_20000.npy'

sys.path.append(VQVAE_DIR)


def get_buffer_size(data):
    num_transitions = 0
    for i in range(len(data)):
        for j in range(len(data[i]['observations'])):
            num_transitions += 1
    return num_transitions


def experiment(variant):
    eval_env = roboverse.make(variant['env'], transpose_image=True)
    expl_env = eval_env
    action_dim = eval_env.action_space.low.size

    if variant['use_robot_state']:
        observation_keys = ['image', 'state']
        state_observation_dim = eval_env.observation_space.spaces['state'].low.size
    else:
        observation_keys = ['image']
        state_observation_dim = 0

    cnn_params = variant['cnn_params']
    cnn_params.update(
        # output_size=action_dim,
        added_fc_input_size=state_observation_dim,
    )

    vqvae = torch.load(variant['vqvae'])
    policy = GaussianVQVAEPolicy(vqvae=vqvae,
                               encoding_type=variant['encoding_type'],
                               max_log_std=0,
                               min_log_std=-6,
                               obs_dim=None,
                               action_dim=action_dim,
                               std_architecture="values",
                               **cnn_params)
    buffer_policy = GaussianVQVAEPolicy(vqvae=vqvae,
                                      encoding_type=variant['encoding_type'],
                                      max_log_std=0,
                                      min_log_std=-6,
                                      obs_dim=None,
                                      action_dim=action_dim,
                                      std_architecture="values",
                                      **cnn_params)

    cnn_params.update(
        output_size=1,
        added_fc_input_size=state_observation_dim + action_dim,
    )

    if variant['use_negative_rewards']:
        cnn_params.update(output_activation=Clamp(max=0))  # rewards are <= 0

    qf1 = ConcatVQVAEWrapper(vqvae, 
        encoding_type=variant['encoding_type'], **cnn_params)
    qf2 = ConcatVQVAEWrapper(vqvae, 
        encoding_type=variant['encoding_type'], **cnn_params)
    target_qf1 = ConcatVQVAEWrapper(vqvae, 
        encoding_type=variant['encoding_type'], **cnn_params)
    target_qf2 = ConcatVQVAEWrapper(vqvae, 
        encoding_type=variant['encoding_type'], **cnn_params)

    with open(variant['buffer'], 'rb') as fl:
        data = np.load(fl, allow_pickle=True)
    num_transitions = get_buffer_size(data)
    max_replay_buffer_size = num_transitions + 10

    image_dim = (48, 48, 3)
    replay_buffer = ObsDictReplayBufferVQVAE(
        vqvae, 
        max_replay_buffer_size,
        expl_env,
        encoding_type=variant['encoding_type'],
        image_dim=image_dim,
        observation_keys=observation_keys
    )
    add_data_to_buffer(data, replay_buffer, observation_keys)

    if variant['use_negative_rewards']:
        if set(np.unique(replay_buffer._rewards)).issubset({0, 1}):
            replay_buffer._rewards = replay_buffer._rewards - 1.0
        assert set(np.unique(replay_buffer._rewards)).issubset({0, -1})

    trainer = AWACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        buffer_policy=buffer_policy,
        **variant['trainer_kwargs']
    )

    # expl_path_collector = None
    # eval_path_collector = None
    eval_policy = MakeDeterministic(policy)
    expl_path_collector = ObsDictPathCollector(
        expl_env,
        policy,
        observation_keys=observation_keys,
    )
    eval_path_collector = ObsDictPathCollector(
        eval_env,
        eval_policy,
        observation_keys=observation_keys,
    )

    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        max_path_length=variant['max_path_length'],
        batch_size=variant['batch_size'],
        num_epochs=variant['num_epochs'],
        num_eval_steps_per_epoch=variant['num_eval_steps_per_epoch'],
        num_expl_steps_per_train_loop=variant['num_expl_steps_per_train_loop'],
        num_trains_per_train_loop=variant['num_trains_per_train_loop'],
        min_num_steps_before_training=variant['min_num_steps_before_training'],
    )

    video_func = VideoSaveFunctionBullet(variant)
    algorithm.post_train_funcs.append(video_func)

    algorithm.to(ptu.device)
    algorithm.train()


def enable_gpus(gpu_str):
    if gpu_str != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default='Widow250MultiTaskGraspShed-v0')
    parser.add_argument("--buffer", type=str, default=BUFFER)

    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument('--use-robot-state', action='store_true', default=False)
    parser.add_argument('--use-negative-rewards', action='store_true',
                        default=False)
    parser.add_argument('--vqvae', type=str, required=True)
    parser.add_argument('--encoding-type', type=str, default='e')
    parser.add_argument("--gpu", default='0', type=str)

    args = parser.parse_args()

    variant = dict(
        algorithm="AWAC-Pixel",

        num_epochs=3000,
        batch_size=256,
        max_path_length=25,
        num_trains_per_train_loop=1000,
        num_eval_steps_per_epoch=125,
        num_expl_steps_per_train_loop=0,
        min_num_steps_before_training=0,

        dump_video_kwargs=dict(
            save_video_period=1,
        ),

        env=args.env,
        buffer=args.buffer,
        use_negative_rewards=args.use_negative_rewards,
        use_robot_state=args.use_robot_state,

        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            beta=args.beta,
            use_automatic_entropy_tuning=False,
            alpha=0,
            compute_bc=False,
            awr_min_q=True,

            bc_num_pretrain_steps=0,
            q_num_pretrain1_steps=0,
            q_num_pretrain2_steps=0,
            # q_num_pretrain2_steps=25000,
            policy_weight_decay=1e-4,
            q_weight_decay=0,

            rl_weight=1.0,
            use_awr_update=True,
            use_reparam_update=False,
            reparam_weight=0.0,
            awr_weight=1.0,
            bc_weight=0.0,

            reward_transform_kwargs=None,
            terminal_transform_kwargs=dict(m=0, b=0),

            awr_use_mle_for_vf=True,
            clip_score=0.5,
        ),
        )

    variant['cnn_params'] = dict(
        input_width=48,
        input_height=48,
        input_channels=3,
        kernel_sizes=[3, 3, 3],
        n_channels=[16, 16, 16],
        strides=[1, 1, 1],
        hidden_sizes=[1024, 512, 256],
        paddings=[1, 1, 1],
        pool_type='max2d',
        pool_sizes=[2, 2, 1],  # the one at the end means no pool
        pool_strides=[2, 2, 1],
        pool_paddings=[0, 0, 0],
        image_augmentation=True,
        image_augmentation_padding=4,
    )
    variant['vqvae'] = args.vqvae
    variant['encoding_type'] = args.encoding_type

    enable_gpus(args.gpu)
    ptu.set_gpu_mode(True)

    exp_prefix = '{}-awac-image-{}'.format(time.strftime("%y-%m-%d"), args.env)
    setup_logger(logger, exp_prefix, LOCAL_LOG_DIR, variant=variant,
                 snapshot_mode='gap_and_last', snapshot_gap=10, )

    experiment(variant)
