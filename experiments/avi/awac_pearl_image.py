import os.path as osp
from collections import OrderedDict
import time
import argparse

import rlkit.torch.pytorch_util as ptu
from rlkit.core import logger
from rlkit.core.meta_rl_algorithm import MetaRLAlgorithm
from rlkit.core.simple_offline_rl_algorithm import OfflineMetaRLAlgorithm
from rlkit.envs.images import GymEnvRenderer
from rlkit.envs.images.plot_renderer import TextRenderer, ScrollingPlotRenderer
# from rlkit.envs.wrappers import NormalizedBoxEnv
# from rlkit.envs.wrappers.flat_to_dict import FlatToDictEnv
from rlkit.misc.asset_loader import load_local_or_remote_file
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.pearl.agent import PEARLAgent
# from rlkit.torch.pearl.diagnostics import (
#     DebugInsertImagesEnv,
#     FlatToDictPearlPolicy,
#     get_env_info_sizes,
# )
from rlkit.torch.networks.cnn import CNN, ConcatCNN
from rlkit.torch.networks import Clamp
from rlkit.torch.pearl.networks import MlpEncoder, MlpDecoder
from rlkit.torch.pearl.launcher_util import load_buffer_onto_algo, EvalPearl
from rlkit.torch.pearl.path_collector import PearlPathCollector
from rlkit.torch.pearl.pearl_cql import PearlCqlTrainer
from rlkit.torch.pearl.sampler import rollout
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic, TanhGaussianPolicyAdapter
from rlkit.launchers.launcher_util import setup_logger
from rlkit.core import logger
from rlkit.data_management.multitask_replay_buffer import ObsDictMultiTaskReplayBuffer
from rlkit.visualization.video import VideoSaveFunctionBasic
import roboverse
import numpy as np
import os
from rlkit.torch.sac.policies import GaussianCNNPolicy

from rlkit.misc.roboverse_utils import add_data_to_buffer_multitask_v2, get_buffer_size_multitask
from rlkit.torch.pearl.pearl_awac import PearlAwacTrainer

CUSTOM_LOG_DIR = '/nfs/kun1/users/avi/doodad-output/'
LOCAL_LOG_DIR = '/media/avi/data/Work/doodad_output/'

BUFFER = '/media/avi/data/Work/github/avisingh599/minibullet/data/test_meta_may26/scripted_Widow250PickPlaceMetaTrainMultiObjectMultiContainer-v0_2021-05-26T15-39-34.npy'


def enable_gpus(gpu_str):
    if gpu_str != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
    return


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
        added_fc_input_size=state_observation_dim + variant['latent_dim'],
    )

    policy = GaussianCNNPolicy(max_log_std=0,
                               min_log_std=-6,
                               obs_dim=None,
                               action_dim=action_dim,
                               std_architecture="values",
                               **cnn_params)
    cnn_params.update(
        output_size=1,
        added_fc_input_size=state_observation_dim + action_dim + variant['latent_dim'],
    )
    if variant['use_negative_rewards']:
        cnn_params.update(output_activation=Clamp(max=0))  # rewards are <= 0
    qf1 = ConcatCNN(**cnn_params)
    qf2 = ConcatCNN(**cnn_params)
    target_qf1 = ConcatCNN(**cnn_params)
    target_qf2 = ConcatCNN(**cnn_params)

    # context encoder
    reward_dim = 1
    latent_dim = variant['latent_dim']
    assert not variant['use_next_obs_in_context']
    context_encoder_output_dim = latent_dim * 2
    cnn_params.update(
        added_fc_input_size=state_observation_dim + action_dim + reward_dim,
        output_size=context_encoder_output_dim,
        hidden_sizes=[256, 256],
        image_augmentation=True
    )
    context_encoder = ConcatCNN(**cnn_params)

    # context decoder (basically a reward predictor)
    cnn_params.update(
        added_fc_input_size=state_observation_dim + action_dim + latent_dim,
        output_size=1,
        hidden_sizes=[256, 256],
        image_augmentation=True,
    )
    context_decoder = ConcatCNN(**cnn_params)
    reward_predictor = context_decoder

    agent = PEARLAgent(
        variant['latent_dim'],
        context_encoder,
        policy,
        reward_predictor,
        obs_keys=observation_keys,
        use_next_obs_in_context=variant['use_next_obs_in_context'],
        _debug_do_not_sqrt=variant['_debug_do_not_sqrt'],
    )

    train_task_indices = list(range(32))
    eval_task_indices = list(range(8))

    pretrain_offline_algo_kwargs = {
        'batch_size': 128,
        'logging_period': 1000,
        'checkpoint_frequency': 10,
        'meta_batch_size': 4,
        'num_batches': int(1e6),  # basically means 1M update steps
        'task_embedding_batch_size': 64,
    }

    with open(variant['buffer'], 'rb') as fl:
        data = np.load(fl, allow_pickle=True)
    num_transitions = get_buffer_size_multitask(data)
    max_replay_buffer_size = num_transitions + 10

    replay_buffer = ObsDictMultiTaskReplayBuffer(
        max_replay_buffer_size,
        expl_env,
        train_task_indices,
        use_next_obs_in_context=variant['use_next_obs_in_context'],
        sparse_rewards=False,
        observation_keys=observation_keys
    )

    enc_replay_buffer = ObsDictMultiTaskReplayBuffer(
        max_replay_buffer_size,
        expl_env,
        train_task_indices,
        use_next_obs_in_context=variant['use_next_obs_in_context'],
        sparse_rewards=False,
        observation_keys=observation_keys
    )

    add_data_to_buffer_multitask_v2(data, replay_buffer, observation_keys)
    add_data_to_buffer_multitask_v2(data, enc_replay_buffer, observation_keys)

    if variant['use_negative_rewards']:
        for ind in train_task_indices:
            task_buffer = replay_buffer.task_buffers[ind]
            if set(np.unique(task_buffer._rewards)).issubset({0, 1}):
                task_buffer._rewards = task_buffer._rewards - 1.0
            assert set(np.unique(task_buffer._rewards)).issubset({0, -1})

        for ind in train_task_indices:
            task_buffer = enc_replay_buffer.task_buffers[ind]
            if set(np.unique(task_buffer._rewards)).issubset({0, 1}):
                task_buffer._rewards = task_buffer._rewards - 1.0
            assert set(np.unique(task_buffer._rewards)).issubset({0, -1})

    trainer_kwargs = variant['trainer_kwargs']
    networks_ignore_context = False
    use_ground_truth_context = False
    trainer = PearlAwacTrainer(
        agent=agent,
        env=expl_env,
        latent_dim=variant['latent_dim'],
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        reward_predictor=reward_predictor,
        context_encoder=context_encoder,
        context_decoder=context_decoder,
        _debug_ignore_context=networks_ignore_context,
        _debug_use_ground_truth_context=use_ground_truth_context,
        **trainer_kwargs
    )

    video_saver = VideoSaveFunctionBasic(variant)

    pretrain_algo = OfflineMetaRLAlgorithm(
        env=eval_env,
        meta_replay_buffer=None,
        replay_buffer=replay_buffer,
        task_embedding_replay_buffer=enc_replay_buffer,
        trainer=trainer,
        train_tasks=train_task_indices,
        eval_tasks=eval_task_indices,
        # extra_eval_fns=[eval_pearl_fn],
        video_saver=video_saver,
        **pretrain_offline_algo_kwargs
    )

    pretrain_algo.to(ptu.device)
    pretrain_algo.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str,
        default='Widow250PickPlaceMetaTestMultiObjectMultiContainer-v0')
    parser.add_argument("--buffer", type=str, default=BUFFER)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument('--use-robot-state', action='store_true', default=False)
    parser.add_argument('--use-negative-rewards', action='store_true',
                        default=False)
    parser.add_argument('--backprop-q-loss-into-encoder', action='store_true',
                        default=False)
    parser.add_argument('--kl-annealing', action='store_true', default=False)
    parser.add_argument("--gpu", default='0', type=str)

    args = parser.parse_args()

    variant = dict(
        algorithm='AWAC-PEARL',
        env=args.env,
        buffer=args.buffer,
        use_negative_rewards=args.use_negative_rewards,
        use_robot_state=args.use_robot_state,
        latent_dim=5,

        use_next_obs_in_context=False,
        _debug_do_not_sqrt=False,

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

            # pearl kwargs
            backprop_q_loss_into_encoder=args.backprop_q_loss_into_encoder,
            train_context_decoder=True,
            kl_annealing = args.kl_annealing,
            kl_annealing_x0=100 * 1000,
            kl_annealing_k=0.01 / 1000,
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

    enable_gpus(args.gpu)
    ptu.set_gpu_mode(True)

    exp_prefix = '{}-awac-pearl-image-{}'.format(time.strftime("%y-%m-%d"), args.env)
    if osp.isdir(CUSTOM_LOG_DIR):
        base_log_dir = CUSTOM_LOG_DIR
    else:
        base_log_dir = LOCAL_LOG_DIR
    setup_logger(logger, exp_prefix, base_log_dir, variant=variant,
                 snapshot_mode='gap_and_last', snapshot_gap=10, )

    experiment(variant)
