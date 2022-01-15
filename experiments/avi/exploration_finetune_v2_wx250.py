import argparse
import time
import os
import gym

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from rlkit.torch.sac.awac_trainer import AWACTrainer
from rlkit.torch.sac.policies import GaussianCNNPolicy, MakeDeterministic
from rlkit.torch.networks.cnn import ConcatCNN

from rlkit.data_management.obs_dict_replay_buffer import ObsDictReplayBuffer
from rlkit.data_management.multitask_replay_buffer import ObsDictMultiTaskReplayBuffer
from rlkit.samplers.data_collector import ObsDictPathCollector, EmbeddingExplorationObsDictPathCollector
from rlkit.launchers.launcher_util import setup_logger
from rlkit.core import logger
from rlkit.torch.networks import Clamp
from rlkit.misc.wx250_utils import (add_multitask_data_to_singletask_buffer_real_robot,
    add_multitask_data_to_multitask_buffer_real_robot, DummyEnv)
from rlkit.exploration_strategies import *
from widowx_envs.widowx.widowx_grasp_env import GraspWidowXResetFreeEnv
from widowx_envs.widowx.env_wrappers import NormalizedBoxEnv
from widowx_envs.utils.params import *

import pickle
import numpy as np
from gym import spaces
from rlkit.launchers.config import LOCAL_LOG_DIR
import torch

import warnings
warnings.filterwarnings("ignore")

def MakeStochastic(policy):
    noise = 0.01
    def new_policy(input):
        out = policy(input)
        out += np.random.normal(noise, size=out.shape)

    return new_policy

def experiment(variant):
    num_tasks = variant['num_tasks']
    eval_env = NormalizedBoxEnv(GraspWidowXResetFreeEnv(
        variant['num_tasks'],
        variant['task_key'],
        variant['object_key'],
        image_save_directory='object_detector_images',
        env_params = {'transpose_image_to_chw': True,
         'wait_time': 0.2,
         'return_full_image': True,
         'override_workspace_boundaries': WORKSPACE_BOUNDARIES,
         'action_mode': '3trans1rot'}
    ))

    """
    Set forward exploration task to be idx num_tasks
    Set reverse exploration task to be idx num_tasks + 1
    """
    variant['exploration_task'] = variant['num_tasks']
    opp_task = variant['exploration_task'] + 1


    if variant['buffer'] != "":
        with open(variant['buffer'], 'rb') as fl:
            data = np.load(fl, allow_pickle=True)
        num_transitions = get_buffer_size(data)
        max_replay_buffer_size = num_transitions + 10
    else:
        num_transitions = int(2e5)
        max_replay_buffer_size = num_transitions + 10

    if variant['use_task_embedding'] and variant['buffer'] != "":
        '''
        Get task embeddings from data
        '''
        num_traj_total = len(data)
        if not 'task_embedding' in data[0]['observations'][0].keys():
            for j in range(num_traj_total):
                for k in range(len(data[j]['observations'])):
                    data[j]['observations'][k]['task_embedding'] = \
                        data[j]['observations'][k]['one_hot_task_id']
                    data[j]['next_observations'][k]['task_embedding'] = \
                        data[j]['next_observations'][k]['one_hot_task_id']
        latent_dim = data[0]['observations'][0]['task_embedding'].shape[0]
        task_embeddings = dict()
        for i in range(variant['num_tasks']):
            task_embeddings[i] = []

        for j in range(num_traj_total):
            task_idx = data[j]['env_infos'][0]['task_idx']
            task_embeddings[task_idx].append(data[j]['observations'][0]['task_embedding'])

        for i in range(variant['num_tasks']):
            task_embeddings[i] = np.asarray(task_embeddings[i])
            task_embeddings[i] = np.mean(task_embeddings[i], axis=0)

        eval_env.observation_space.spaces.update(
            {'task_embedding': spaces.Box(
                low=np.array([-100] * latent_dim),
                high=np.array([100] * latent_dim),
            )})
        eval_env = EmbeddingWrapper(eval_env, embeddings=task_embeddings)

        # Get task embeddings from data for exploration strategy
        for i in range(len(data)):
            for j in range(len(data[i]['observations'])):
                task_embeddings_batch.append(data[i]['observations'][j]['task_embedding'])
        task_embeddings_batch = np.array(task_embeddings_batch)
    else:
        '''
        Use one-hot task embeddings with dim num_tasks
        '''
        eval_env.observation_space.spaces.update(
            {'task_embedding': spaces.Box(
                low=np.array([0] * num_tasks),
                high=np.array([1] * num_tasks),
            )})

        # Task embeddings are just 1 hot vectors (for exploration strategy)
        task_embeddings_batch = np.zeros((args.num_tasks, 1, args.num_tasks))
        for i in range(args.num_tasks):
            one_hot_embedding = np.zeros(args.num_tasks)
            one_hot_embedding[i] = 1
            task_embeddings_batch[i][0] = one_hot_embedding

    expl_env = eval_env
    expl_env.reset()

    action_dim = eval_env.action_space.low.size
    if variant['use_robot_state']:
        observation_keys = ['image', 'state', 'task_embedding']
        state_observation_dim = eval_env.observation_space.spaces['state'].low.size
    else:
        observation_keys = ['image', 'task_embedding']
        state_observation_dim = 0

    cnn_params = variant['cnn_params']
    if variant['use_task_embedding']:
        cnn_params.update(added_fc_input_size=state_observation_dim + latent_dim)
    else:
        cnn_params.update(added_fc_input_size=state_observation_dim + num_tasks)

    policy = GaussianCNNPolicy(max_log_std=0,
                               min_log_std=-6,
                               obs_dim=None,
                               action_dim=action_dim,
                               std_architecture="values",
                               **cnn_params)
    #policy = MakeStochastic(policy)
    buffer_policy = GaussianCNNPolicy(max_log_std=0,
                                      min_log_std=-6,
                                      obs_dim=None,
                                      action_dim=action_dim,
                                      std_architecture="values",
                                      **cnn_params)
    cnn_params.update(
        output_size=1,
    )
    if variant['use_task_embedding']:
        cnn_params.update(
            added_fc_input_size=state_observation_dim + latent_dim + action_dim,
        )
    else:
        cnn_params.update(
            added_fc_input_size=state_observation_dim + num_tasks + action_dim,
        )
    if variant['use_negative_rewards']:
        cnn_params.update(output_activation=Clamp(max=0))  # rewards are <= 0

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


    # we need to add room for an exploration tasks
    num_buffer_tasks = num_tasks * 2

    if variant['buffer'] != "":
        with open(variant['buffer'], 'rb') as f:
            replay_buffer = pickle.load(f)
    else:
        # create new replay buffer
        replay_buffer = ObsDictMultiTaskReplayBuffer(
            max_replay_buffer_size,
            expl_env,
            np.arange(num_buffer_tasks),
            use_next_obs_in_context=False,
            sparse_rewards=False,
            observation_keys=observation_keys
        )
        if variant['buffers'] != []:
            buffer_params = {task: b for task, b in enumerate(variant['buffers'])}
            add_multitask_data_to_multitask_buffer_real_robot(buffer_params, replay_buffer,
                task_encoder=None, embedding_mode=variant['embedding_mode'],
                encoder_type=variant['task_encoder_type'])

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
        #use_reward_as_terminal=True,
        **variant['trainer_kwargs']
    )

    if variant['exploration_strategy'] == 'gaussian':
        exploration_strategy = GaussianExplorationStrategy(task_embeddings_batch, policy=eval_policy,
            q_function=qf1, n_components=10)
    elif variant['exploration_strategy'] == 'gaussian_filtered':
        exploration_strategy = GaussianExplorationStrategy(task_embeddings_batch, policy=eval_policy,
            q_function=qf1, n_components=10)
    elif variant['exploration_strategy'] == 'cem':
        exploration_strategy = CEMExplorationStrategy(task_embeddings_batch,
            update_frequency=variant['exploration_update_frequency'], n_components=num_tasks)
    elif variant['exploration_strategy'] == 'closest':
        exploration_strategy = ClosestExplorationStrategy(task_embeddings_batch,
        exploration_period=variant['closest_expl_period'])
    elif variant['exploration_strategy'] == 'fast':
        exploration_strategy = FastExplorationStrategy(task_embeddings_batch,
            update_frequency=variant['exploration_update_frequency'], n_components=10)
    else:
        raise NotImplementedError

    if args.finetuning_checkpoint != "":
        with open(args.finetuning_checkpoint, "rb") as f:
            data = torch.load(f)
            exploration_strategy = data['exploration/exploration_strategy']
            policy = data['trainer/policy']
            trainer.policy = policy
            trainer.buffer_policy = data['trainer/buffer_policy']
            trainer.qf1 = data['trainer/qf1']
            trainer.target_qf1 = data['trainer/target_qf1']
            trainer.qf2 = data['trainer/qf2']
            trainer.target_qf2 = data['trainer/target_qf2']

    eval_policy = MakeDeterministic(policy)
    expl_path_collector = EmbeddingExplorationObsDictPathCollector(
        exploration_strategy,
        expl_env,
        policy,
        observation_keys=observation_keys,
        expl_reset_free=args.expl_reset_free,
        epochs_per_reset=variant['epochs_per_reset'],
        exploration_task=variant['exploration_task'],
        relabel_rewards=True
    )
    eval_path_collector = EmbeddingExplorationObsDictPathCollector(
        exploration_strategy,
        expl_env,
        policy,
        observation_keys=observation_keys,
        expl_reset_free=False,
        epochs_per_reset=variant['epochs_per_reset'],
        exploration_task=variant['exploration_task'],
        do_cem_update=False,
        relabel_rewards=True
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
        meta_batch_size=variant['meta_batch_size'],
        num_epochs=variant['num_epochs'],
        num_eval_steps_per_epoch=variant['num_eval_steps_per_epoch'],
        num_expl_steps_per_train_loop=variant['num_expl_steps_per_train_loop'],
        num_trains_per_train_loop=variant['num_trains_per_train_loop'],
        min_num_steps_before_training=variant['min_num_steps_before_training'],
        multi_task=True,
        exploration_task=variant['exploration_task'],
        train_tasks=[variant['exploration_task'], opp_task],
        eval_tasks=[variant['exploration_task'], opp_task],
        log_keys_to_remove=["exploration/env", "evaluation/env"],
        save_exploration_paths=True
    )

    # TODO: Add video saving functionality
    #video_func = VideoSaveFunctionBullet(variant)
    #algorithm.post_train_funcs.append(video_func)

    algorithm.to(ptu.device)
    algorithm.train()


def enable_gpus(gpu_str):
    if gpu_str != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", type=str, required=True)
    parser.add_argument("-fc", "--finetuning_checkpoint", type=str, default="")
    parser.add_argument("-d", "--discount", type=float, default=0.95)
    parser.add_argument("--clip-score", type=float, default=5.0)
    parser.add_argument("--num-tasks", type=int, default=32)
    parser.add_argument("--task-key", type=str, default='pinktrayleft', choices=('pinktrayleft',
        'redplateright'))
    parser.add_argument("--object-key", type=str, required=True)
    parser.add_argument("--obs-dict-buffer", type=str, default="")
    parser.add_argument("--buffers", type=str, nargs='+')
    parser.add_argument("--buffer-variant", type=str, default="")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument('--use-robot-state', action='store_true', default=False)
    parser.add_argument('--use-negative-rewards', action='store_true',
                        default=False)
    parser.add_argument('--expl-reset-free', action='store_true', default=False)
    parser.add_argument("--task-encoder", default="", type=str)
    parser.add_argument("--encoder-type", default='image', choices=('image', 'trajectory'))
    parser.add_argument("--embedding-mode", type=str, choices=('one-hot', 'single', 'batch'), required=True)
    parser.add_argument('-e', '--exploration-strategy', type=str,
        choices=('gaussian', 'gaussian_filtered', 'cem', 'fast', 'closest'))
    parser.add_argument("--gpu", default='0', type=str)

    args = parser.parse_args()

    assert args.buffer_variant != "" or args.buffers != "" or args.obs_dict_buffer != ""
    buffers = []
    if args.buffers:
        buffers = set()
        for buffer_path in args.buffers:
            if '.pkl' in buffer_path or '.npy' in buffer_path:
                buffers.add(buffer_path)
            else:
                path = Path(buffer_path)
                buffers.update(list(path.rglob('*.pkl')))
                buffers.update(list(path.rglob('*.npy')))
        buffers = [str(b) for b in buffers]
        if args.buffer_variant:
            '''Use buffer variant to get ordered list of buffer paths'''
            import json
            buffer_variant = open(args.buffer_variant)
            data = json.load(buffer_variant)
            buffer_order = data["buffers"]
            new_buffers = []
            for i in buffer_order:
                buffer_name = os.path.basename(i)
                for j in buffers:
                    if buffer_name in j:
                        new_buffers.append(j)
                        break
            buffers = new_buffers
    print(buffers)
    variant = dict(
        algorithm="AWAC-Pixel",

        num_epochs=3000,
        batch_size=64,
        meta_batch_size=4,
        max_path_length=20,
        num_trains_per_train_loop=1000,
        num_eval_steps_per_epoch=0,
        num_expl_steps_per_train_loop=10 * 20,
        min_num_steps_before_training=10 * 20,

        dump_video_kwargs=dict(
            save_video_period=1,
        ),

        num_tasks=args.num_tasks,
        task_key=args.task_key,
        object_key=args.object_key,
        checkpoint=args.checkpoint,
        finetuning_checkpoint=args.finetuning_checkpoint,
        buffer=args.obs_dict_buffer,
        buffers=buffers,
        use_negative_rewards=args.use_negative_rewards,
        use_robot_state=args.use_robot_state,
        use_task_embedding=False,

        exploration_task = args.num_tasks,
        exploration_strategy = args.exploration_strategy,
        exploration_update_frequency=10,
        expl_reset_free = args.expl_reset_free,
        closest_expl_period = 15,
        epochs_per_reset = 0,

        trainer_kwargs=dict(
            discount=args.discount,
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
            clip_score=args.clip_score,
        ),

        task_encoder_checkpoint=args.task_encoder,
        task_encoder_type=args.encoder_type,
        embedding_mode=args.embedding_mode,
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

    exp_prefix = '{}-exploration-awac-image-real-robot'.format(time.strftime("%y-%m-%d"))
    setup_logger(logger, exp_prefix, LOCAL_LOG_DIR, variant=variant,
                 snapshot_mode='gap_and_last', snapshot_gap=1, )

    experiment(variant)
