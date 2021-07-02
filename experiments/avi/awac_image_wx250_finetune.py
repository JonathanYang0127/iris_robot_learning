
import argparse
import time
import os.path as osp
import os

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from rlkit.torch.sac.awac_trainer import AWACTrainer
from rlkit.torch.sac.policies import GaussianCNNPolicy, MakeDeterministic
from rlkit.torch.networks.cnn import CNN, ConcatCNN

from rlkit.data_management.obs_dict_replay_buffer import ObsDictReplayBuffer
from rlkit.samplers.data_collector import MdpPathCollector, ObsDictPathCollector
from rlkit.launchers.launcher_util import setup_logger
from rlkit.core import logger
from rlkit.torch.networks import Clamp
from rlkit.misc.roboverse_utils import add_data_to_buffer, VideoSaveFunctionBullet
from rlkit.misc.wx250_utils import add_data_to_buffer_real_robot, DummyEnv
from rlkit.envs.wrappers import NormalizedBoxEnv
from widowx_envs.widowx.widowx_grasp_env import GraspWidowXEnv
from widowx_envs.utils.params_private import *
from widowx_envs.utils.object_detection.object_detector_dl import ObjectDetectorDL

# import roboverse
import numpy as np
import torch

from rlkit.launchers.config import LOCAL_LOG_DIR

def get_buffer_size(data):
    num_transitions = 0
    for i in range(len(data)):
        for j in range(len(data[i]['observations'])):
            num_transitions += 1
    return num_transitions


def experiment(variant):
    image_size = 64
    eval_env = NormalizedBoxEnv(GraspWidowXEnv(
        {'transpose_image_to_chw': True,
         'wait_time': 0.2,
         'return_full_image': True,
         'override_workspace_boundaries': WORKSPACE_BOUNDARIES,
         'action_mode': '3trans'}
    ))

    object_detector = ObjectDetectorDL(env=eval_env, save_dir=".")

    print("eval_env.action_space.low", eval_env.action_space.low)
    print("eval_env.action_space.high", eval_env.action_space.high)
    print("eval_env._adim", eval_env._adim)
    print("eval_env._sdim", eval_env._sdim)
    expl_env = eval_env
    action_dim = eval_env.action_space.low.size

    if variant['use_robot_state']:
        observation_keys = ['image', 'state']
        state_observation_dim = eval_env.observation_space.spaces['state'].low.size
    else:
        observation_keys = ['image']
        state_observation_dim = 0

    if args.checkpoint != '':
        with open(variant['checkpoint'], 'rb') as f:
            checkpoint = torch.load(f)
        # import ipdb; ipdb.set_trace()
        policy = checkpoint['exploration/policy']
        buffer_policy = checkpoint['trainer/buffer_policy']
        eval_policy = checkpoint['evaluation/policy']
    else:
        print("No Checkpoint Specified! Initializing policy and trainer")

        cnn_params = variant['cnn_params']
        cnn_params.update(
            # output_size=action_dim,
            added_fc_input_size=state_observation_dim,
        )

        policy = GaussianCNNPolicy(max_log_std=0,
                                   min_log_std=-6,
                                   obs_dim=None,
                                   action_dim=action_dim,
                                   std_architecture="values",
                                   **cnn_params)
        buffer_policy = GaussianCNNPolicy(max_log_std=0,
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

    replay_buffer = ObsDictReplayBuffer(
        int(1E6),
        expl_env,
        observation_keys=observation_keys
    )
    add_data_to_buffer_real_robot(variant['buffer'], replay_buffer,
                       validation_replay_buffer=None,
                       validation_fraction=0.8, num_trajs_limit=variant['num_trajs_limit'])

    if variant['use_negative_rewards']:
        if set(np.unique(replay_buffer._rewards)).issubset({0, 1}):
            replay_buffer._rewards = replay_buffer._rewards - 1.0
        assert set(np.unique(replay_buffer._rewards)).issubset({0, -1})

    if args.checkpoint != '':
        qf1 = checkpoint['trainer/qf1']
        qf2 = checkpoint['trainer/qf2']
        target_qf1 = checkpoint['trainer/target_qf1']
        target_qf2 = checkpoint['trainer/target_qf2']
    else:
        qf1 = ConcatCNN(**cnn_params)
        qf2 = ConcatCNN(**cnn_params)
        target_qf1 = ConcatCNN(**cnn_params)
        target_qf2 = ConcatCNN(**cnn_params)

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
        log_keys_to_remove=['evaluation/env', 'exploration/env'],
        object_detector=object_detector,
    )

    video_func = VideoSaveFunctionBullet(variant)
    algorithm.post_train_funcs.append(video_func)

    algorithm.to(ptu.device)

    if variant['use_bc']:
        trainer.pretrain_policy_with_bc(
            policy,
            replay_buffer,
            1000 * variant['num_epochs'],
        )
    else:
        algorithm.train()


def enable_gpus(gpu_str):
    if gpu_str != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer", type=str, required=True)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument('--use-robot-state', action='store_true', default=False)
    parser.add_argument('--use-negative-rewards', action='store_true',
                        default=False)
    parser.add_argument("--gpu", default='0', type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--use-bc", action="store_true", default=False)
    parser.add_argument("--num-trajs-limit", default=0, type=int)
    parser.add_argument("--checkpoint", type=str, default='')
    args = parser.parse_args()

    alg = 'BC' if args.use_bc else 'AWAC-Pixel'

    variant = dict(
        algorithm=alg,

        num_epochs=3000,
        batch_size=256,
        max_path_length=15,
        num_trains_per_train_loop=1000,
        num_eval_steps_per_epoch=0,
        num_expl_steps_per_train_loop=30,
        min_num_steps_before_training=0,

        dump_video_kwargs=dict(
            save_video_period=1,
        ),

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
            bc_weight=1.0,

            reward_transform_kwargs=None,
            terminal_transform_kwargs=dict(m=0, b=0),

            awr_use_mle_for_vf=True,
            clip_score=0.5,
        ),
        )

    variant['cnn_params'] = dict(
        input_width=64,
        input_height=64,
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
    variant['seed'] = args.seed
    variant['use_bc'] = args.use_bc
    if args.num_trajs_limit > 0:
        variant['num_trajs_limit'] = args.num_trajs_limit
    else:
        variant['num_trajs_limit'] = None
    variant['checkpoint'] = args.checkpoint

    enable_gpus(args.gpu)
    ptu.set_gpu_mode(True)

    exp_prefix = '{}-{}-wx250-finetune'.format(time.strftime("%y-%m-%d"), alg)
    setup_logger(logger, exp_prefix, LOCAL_LOG_DIR, variant=variant,
                 snapshot_mode='gap_and_last', snapshot_gap=10, seed=args.seed)

    experiment(variant)