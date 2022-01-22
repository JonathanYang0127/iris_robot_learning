import argparse
import time
import os
import gym
from rlkit.core.roboverse_serializable import Serializable

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithmRnd
from rlkit.torch.sac.awac_trainer import AWACTrainer
from rlkit.torch.sac.policies import GaussianCNNPolicy, MakeDeterministic, AddNoise
from rlkit.torch.networks.cnn import ConcatCNN

from rlkit.data_management.obs_dict_replay_buffer import ObsDictReplayBuffer
from rlkit.data_management.multitask_replay_buffer import ObsDictMultiTaskReplayBuffer
from rlkit.samplers.data_collector import RndPathCollector
from rlkit.launchers.launcher_util import setup_logger
from rlkit.core import logger
from rlkit.torch.networks import Clamp
from rlkit.misc.roboverse_utils import add_multitask_data_to_singletask_buffer_v2, \
    add_multitask_data_to_multitask_buffer_v2, \
    VideoSaveFunctionBullet, get_buffer_size, add_data_to_buffer
from rlkit.exploration_strategies import *

import roboverse
import numpy as np
from gym import spaces
from rlkit.launchers.config import LOCAL_LOG_DIR
import torch

def experiment(variant):
    num_tasks = variant['num_tasks']
    env_num_tasks = num_tasks
    if args.reset_free:
        #hacky change because the num_tasks passed into roboverse doesn't count exploration
        env_num_tasks //= 2
    eval_env = roboverse.make(variant['env'], transpose_image=True, num_tasks=env_num_tasks)
    latent_dim = num_tasks + 1 # add one extra index for online data
    if variant['exploration_task'] < num_tasks:
        if variant['exploration_task'] < env_num_tasks:
            opp_task = variant['exploration_task']+env_num_tasks
        else:
            opp_task = variant['exploration_task']-env_num_tasks
    else:
        if variant['exploration_task'] < num_tasks + env_num_tasks:
            opp_task = variant['exploration_task']+env_num_tasks
        else:
            opp_task = variant['exploration_task']-env_num_tasks

    if variant['buffer'] is not None:
        with open(variant['buffer'], 'rb') as fl:
            data = np.load(fl, allow_pickle=True)
        num_transitions = get_buffer_size(data)
        max_replay_buffer_size = num_transitions + 10

        num_traj_total = len(data)
        if not 'task_embedding' in data[0]['observations'][0].keys():
            for j in range(num_traj_total):
                for k in range(len(data[j]['observations'])):
                    # add an extra expl index
                    data[j]['observations'][k]['task_embedding'] = \
                        np.concatenate([data[j]['observations'][k]['one_hot_task_id'], np.array([0.])])
                    data[j]['next_observations'][k]['task_embedding'] = \
                        np.concatenate([data[j]['next_observations'][k]['one_hot_task_id'], np.array([0.])])
    else:
        num_transitions = 8192*30
        max_replay_buffer_size = num_transitions + 10

    task_embeddings = dict()
    for i in range(variant['num_tasks']):
        one_hot_embedding = np.zeros(variant['num_tasks'])
        one_hot_embedding[i] = 1
        task_embeddings[i] = one_hot_embedding

    eval_env.observation_space.spaces.update(
        {'task_embedding': spaces.Box(
            low=np.array([-100] * latent_dim),
            high=np.array([100] * latent_dim),
        )})
    eval_env = EmbeddingWrapper(eval_env, embeddings=task_embeddings)

    expl_env = eval_env
    expl_env.reset_task(variant['exploration_task'])
    expl_env.reset()

    action_dim = eval_env.action_space.low.size
    observation_keys = ['image',]

    observation_keys.append('task_embedding')

    if variant['use_robot_state']:
        observation_keys.append('state')
        state_observation_dim = eval_env.observation_space.spaces['state'].low.size
    else:
        state_observation_dim = 0

    cnn_params = variant['cnn_params']
    cnn_params.update(added_fc_input_size=state_observation_dim + latent_dim)

    policy = GaussianCNNPolicy(max_log_std=0,
                               min_log_std=-6,
                               obs_dim=None,
                               action_dim=action_dim,
                               std_architecture="values",
                               **cnn_params)
    perturb_policy = GaussianCNNPolicy(max_log_std=0,
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
    perturb_buffer_policy = GaussianCNNPolicy(max_log_std=0,
                                      min_log_std=-6,
                                      obs_dim=None,
                                      action_dim=action_dim,
                                      std_architecture="values",
                                      **cnn_params)
    rnd_model = RNDModel(**cnn_params, output_size=variant['rnd_output_size'])
    
    cnn_params.update(
        output_size=1,
    )
    cnn_params.update(
        added_fc_input_size=state_observation_dim + latent_dim + action_dim,
    )
    if variant['use_negative_rewards']:
        cnn_params.update(output_activation=Clamp(max=0))  # rewards are <= 0

    qf1 = ConcatCNN(**cnn_params)
    qf2 = ConcatCNN(**cnn_params)
    target_qf1 = ConcatCNN(**cnn_params)
    target_qf2 = ConcatCNN(**cnn_params)
    perturb_qf1 = ConcatCNN(**cnn_params)
    perturb_qf2 = ConcatCNN(**cnn_params)
    perturb_target_qf1 = ConcatCNN(**cnn_params)
    perturb_target_qf2 = ConcatCNN(**cnn_params)

    if args.checkpoint is not None:
        ext = os.path.splitext(args.checkpoint)[-1]
        with open(args.checkpoint, 'rb') as handle:
            if ext == ".pt":
                params = torch.load(handle)
                policy = params['trainer/policy']
                eval_policy = MakeDeterministic(policy)
                qf1 = params['trainer/qf1']
                qf2 = params['trainer/qf2']
                target_qf1 = params['trainer/target_qf1']
                target_qf2 = params['trainer/target_qf2']
            elif ext == ".pkl":
                policy = pickle.load(handle)
                eval_policy = MakeDeterministic(policy)


    # allocate buffers for all test tasks 
    num_buffer_tasks = num_tasks * 2
    replay_buffer = ObsDictMultiTaskReplayBuffer(
        max_replay_buffer_size,
        expl_env,
        np.arange(num_buffer_tasks),
        use_next_obs_in_context=False,
        sparse_rewards=False,
        observation_keys=observation_keys
    )

    if variant['buffer'] is not None:
        add_multitask_data_to_multitask_buffer_v2(data, replay_buffer,
                                            observation_keys, num_tasks)
    # if variant['exploration_task'] < num_tasks:
    #     replay_buffer.task_buffers[variant['exploration_task']].bias_point = replay_buffer.task_buffers[variant['exploration_task']]._top
    #     replay_buffer.task_buffers[variant['exploration_task']].before_bias_point_probability = 0.3
    #     replay_buffer.task_buffers[opp_task].bias_point = replay_buffer.task_buffers[opp_task]._top
    #     replay_buffer.task_buffers[opp_task].before_bias_point_probability = 0.3

    # if len(data[0]['observations'][0]['image'].shape) > 1:
    #     add_data_to_buffer(data, replay_buffer, observation_keys)
    # else:
    #     add_data_to_buffer_new(data, replay_buffer, observation_keys)

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
        multitask=True,
        **variant['trainer_kwargs']
    )
    perturb_trainer = AWACTrainer(
        env=eval_env,
        policy=perturb_policy,
        qf1=perturb_qf1,
        qf2=perturb_qf2,
        target_qf1=perturb_target_qf1,
        target_qf2=perturb_target_qf2,
        buffer_policy=perturb_buffer_policy,
        multitask=True,
        rnd=rnd_model,
        **variant['trainer_kwargs']
    )

    eval_policy = MakeDeterministic(policy)
    expl_path_collector = RndPathCollector(
        expl_env,
        policy,
        perturb_policy,
        observation_keys=observation_keys,
        epochs_per_reset=variant['epochs_per_reset'],
        exploration_task=variant['exploration_task'],
        latent_dim=latent_dim,
        expl_reset_free=True
    )
    eval_path_collector = RndPathCollector(
        expl_env,
        eval_policy,
        perturb_policy,
        observation_keys=observation_keys,
        epochs_per_reset=variant['epochs_per_reset'],
        exploration_task=variant['exploration_task'],
        latent_dim=latent_dim,
        expl_reset_free=False,
    )

    if args.buffer is None:
        train_tasks = [] # batch rl alg add expl tasks to train tasks
    else:
        train_tasks = np.arange(num_tasks)
    
    algorithm = TorchBatchRLAlgorithmRnd(
        trainer=trainer,
        perturb_trainer=perturb_trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        max_path_length=variant['max_path_length'],
        batch_size=variant['batch_size'],
        meta_batch_size=variant['meta_batch_size'],
        num_epochs=variant['num_epochs'],
        num_eval_steps_per_epoch=variant['num_eval_steps_per_epoch'],
        num_expl_steps_per_train_loop=variant['num_expl_steps_per_train_loop'],
        num_trains_per_train_loop=variant['num_trains_per_train_loop'],
        min_num_steps_before_training=variant['min_num_steps_before_training'],
        multi_task=True,
        exploration_task=variant['exploration_task'],
        train_tasks=train_tasks,
        eval_tasks=[variant['exploration_task']],
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
    parser.add_argument("--env", type=str, default='Widow250PickPlaceMetaTrainMultiObjectMultiContainer-v0')
    parser.add_argument("-c", "--checkpoint", type=str)
    parser.add_argument("--num-tasks", type=int, default=32)
    parser.add_argument("--exploration-task", type=int)
    parser.add_argument("--buffer", type=str)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument('--use-robot-state', action='store_true', default=False)
    parser.add_argument('--use-negative-rewards', action='store_true',
                        default=False)
    parser.add_argument('--reset-free', action='store_true', default=False)
    parser.add_argument('--expl-reset-free', action='store_true', default=False)
    parser.add_argument("--gpu", default='0', type=str)
    parser.add_argument("--seed", default=0, type=int)

    args = parser.parse_args()

    variant = dict(
        algorithm="AWAC-Pixel",

        num_epochs=3000,
        batch_size=64,
        meta_batch_size=4,
        max_path_length=40,
        num_trains_per_train_loop=10,
        num_eval_steps_per_epoch=2 * 40,
        num_expl_steps_per_train_loop=2 * 40,
        min_num_steps_before_training=0,

        dump_video_kwargs=dict(
            save_video_period=1,
        ),

        env=args.env,
        num_tasks=args.num_tasks,
        checkpoint=args.checkpoint,
        buffer=args.buffer,
        use_negative_rewards=args.use_negative_rewards,
        use_robot_state=args.use_robot_state,
        use_task_embedding=True,
        seed=args.seed,

        exploration_task = args.exploration_task,
        exploration_update_frequency=10,
        expl_reset_free = args.expl_reset_free,
        epochs_per_reset = 1,
        closest_expl_period = 15,
        expl_policy_noise = 0.0,
        rnd_output_size = 5,

        trainer_kwargs=dict(
            discount=0.9666,
            use_reward_as_terminal=False,
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

    enable_gpus(args.gpu)
    ptu.set_gpu_mode(True)

    # random_num = np.random.randint(10000)
    # exp_prefix = '{}-exploration-awac-image-{}-{}'.format(time.strftime("%y-%m-%d"), args.env, random_num)
    exp_prefix = '{}-exploration-awac-image-{}'.format(time.strftime("%y-%m-%d"), args.env)
    setup_logger(logger, exp_prefix, LOCAL_LOG_DIR, variant=variant,
                 snapshot_mode='gap_and_last', snapshot_gap=10, seed=args.seed)

    experiment(variant)
