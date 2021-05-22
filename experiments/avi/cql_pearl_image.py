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

CUSTOM_LOG_DIR = '/nfs/kun1/users/avi/doodad-output/'
LOCAL_LOG_DIR = '/media/avi/data/Work/doodad_output/'

BUFFER_1 = '/media/avi/data/Work/github/avisingh599/minibullet/data/may14_meta_Widow250MultiTaskGraspShed-v0_1000_save_all_noise_0.1_2021-05-14T16-27-16/may14_meta_Widow250MultiTaskGraspShed-v0_1000_save_all_noise_0.1_2021-05-14T16-27-16_1000.npy'
BUFFER_2 = '/media/avi/data/Work/github/avisingh599/minibullet/data/may14_meta_Widow250MultiTaskGraspVase-v0_1000_save_all_noise_0.1_2021-05-14T16-39-22/may14_meta_Widow250MultiTaskGraspVase-v0_1000_save_all_noise_0.1_2021-05-14T16-39-22_1000.npy'


def process_keys(observations, observation_keys):
    output = []
    for i in range(len(observations)):
        observation = dict()
        for key in observation_keys:
            if key == 'image':
                image = observations[i]['image']
                if len(image.shape) == 3:
                    image = np.transpose(image, [2, 0, 1])
                    image = (image.flatten())/255.0
                else:
                    print('image shape: {}'.format(image.shape))
                    raise ValueError
                observation[key] = image
            elif key == 'state':
                observation[key] = np.array(observations[i]['state'])
            else:
                raise NotImplementedError
        output.append(observation)
    return output


def add_data_to_buffer(data, replay_buffer, observation_keys, task):

    for j in range(len(data)):
        assert (len(data[j]['actions']) == len(data[j]['observations']) == len(
            data[j]['next_observations']))

        path = dict(
            rewards=[np.asarray([r]) for r in data[j]['rewards']],
            actions=data[j]['actions'],
            terminals=[np.asarray([t]) for t in data[j]['terminals']],
            observations=process_keys(data[j]['observations'], observation_keys),
            next_observations=process_keys(
                data[j]['next_observations'], observation_keys),
        )
        replay_buffer.add_path(task, path)


def experiment(variant):
    eval_env = roboverse.make(variant['env'], transpose_image=True)
    expl_env = eval_env
    action_dim = eval_env.action_space.low.size

    latent_dim = variant['latent_dim']
    # Q-functions
    state_observation_dim = 0
    if variant['use_robot_state']:
        print(eval_env.observation_space)
        state_observation_dim = eval_env.observation_space.spaces[
            'state'].low.size
    cnn_params = variant['cnn_params']
    cnn_params.update(
        input_width=48,
        input_height=48,
        input_channels=3,
        output_size=1,
        added_fc_input_size=state_observation_dim + action_dim + latent_dim,
    )
    qf1 = ConcatCNN(**cnn_params)
    qf2 = ConcatCNN(**cnn_params)
    target_qf1 = ConcatCNN(**cnn_params)
    target_qf2 = ConcatCNN(**cnn_params)

    # policy
    if variant['use_robot_state']:
        policy_added_input_dim = state_observation_dim
    else:
        policy_added_input_dim = 0
    cnn_params.update(
        output_size=256,
        added_fc_input_size=policy_added_input_dim,
        hidden_sizes=[1024, 512],
    )
    policy_obs_processor = CNN(**cnn_params)
    policy = TanhGaussianPolicyAdapter(
        policy_obs_processor,
        cnn_params['output_size'],
        action_dim,
        hidden_sizes=[256, 256, 256],
    )

    if variant['use_robot_state']:
        observation_keys = ['image', 'state']
    else:
        observation_keys = ['image']

    # context encoder
    reward_dim = 1
    assert not variant['use_next_obs_in_context']
    context_encoder_output_dim = latent_dim * 2
    cnn_params.update(
        added_fc_input_size=state_observation_dim + action_dim + reward_dim,
        output_size=context_encoder_output_dim,
        hidden_sizes=[256, 256],
    )
    context_encoder = ConcatCNN(**cnn_params)

    # context decoder (basically a reward predictor)
    cnn_params.update(
        added_fc_input_size=state_observation_dim + action_dim + latent_dim,
        output_size=1,
        hidden_sizes=[256, 256],
        image_augmentation=False,
    )
    context_decoder = ConcatCNN(**cnn_params)
    reward_predictor = context_decoder

    agent = PEARLAgent(
        latent_dim,
        context_encoder,
        policy,
        reward_predictor,
        obs_keys=observation_keys,
        use_next_obs_in_context=variant['use_next_obs_in_context'],
        _debug_do_not_sqrt=variant['_debug_do_not_sqrt'],
    )

    trainer_kwargs = variant['trainer_kwargs']
    trainer = PearlCqlTrainer(
        latent_dim=variant['latent_dim'],
        agent=agent,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        reward_predictor=reward_predictor,
        context_encoder=context_encoder,
        context_decoder=context_decoder,
        action_space=expl_env.action_space,
        **trainer_kwargs
    )

    train_task_indices = [0, 1]
    eval_task_indices = [0, 1]

    pretrain_offline_algo_kwargs = {
        'batch_size': 128,
        'logging_period': 1000,
        'checkpoint_frequency': 10,
        'meta_batch_size': 4,
        'num_batches': int(1e6),  # basically means 1M update steps
        'task_embedding_batch_size': 64,
    }

    max_replay_buffer_size = int(5E5)

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

    # import IPython; IPython.embed()

    with open(variant['buffer_a'], 'rb') as fl:
        data = np.load(fl, allow_pickle=True)
    add_data_to_buffer(data, replay_buffer, observation_keys, task=0)
    add_data_to_buffer(data, enc_replay_buffer, observation_keys, task=0)

    with open(variant['buffer_b'], 'rb') as fl:
        data = np.load(fl, allow_pickle=True)
    add_data_to_buffer(data, replay_buffer, observation_keys, task=1)
    add_data_to_buffer(data, enc_replay_buffer, observation_keys, task=1)


    # eval_pearl_fn = EvalPearl(
    #     algorithm, train_task_indices, eval_task_indices
    # )
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

    # algorithm.to(ptu.device)
    # algorithm.train()


def enable_gpus(gpu_str):
    if gpu_str != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
    return


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algorithm="Pearl-CQL",
        latent_dim=5,
        dump_video_kwargs=dict(
            save_video_period=1,
        ),
        # from standard image-based CQL
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            policy_lr=1E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,

            # Target nets/ policy vs Q-function update
            policy_eval_start=0,
            num_qs=2,

            # min Q
            temp=1.0,
            min_q_version=3,
            min_q_weight=1.0,

            # lagrange
            with_lagrange=False,  # Defaults to False
            lagrange_thresh=10.0,

            # extra params
            num_random=1,
            max_q_backup=False,
            deterministic_backup=True,

            # pearl kwargs
            backprop_q_loss_into_encoder=False,
            train_context_decoder=True,
        ),
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default='Widow250MetaGraspVaseShed-v0')
    parser.add_argument("--buffer-a", type=str, default=BUFFER_1)
    parser.add_argument("--buffer-b", type=str, default=BUFFER_2)

    parser.add_argument('--use-robot-state', action='store_true', default=False)
    parser.add_argument("--gpu", default='0', type=str)
    args = parser.parse_args()

    variant['env'] = args.env
    variant['use_robot_state'] = args.use_robot_state
    variant['buffer_a'] = args.buffer_a
    variant['buffer_b'] = args.buffer_b

    variant['_debug_do_not_sqrt'] = False
    variant['use_next_obs_in_context'] = False

    variant['cnn_params'] = dict(
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

    exp_prefix = '{}-pearl-cql-{}'.format(time.strftime("%y-%m-%d"), args.env)
    if osp.isdir(CUSTOM_LOG_DIR):
        base_log_dir = CUSTOM_LOG_DIR
    else:
        base_log_dir = LOCAL_LOG_DIR
    setup_logger(logger, exp_prefix, base_log_dir, variant=variant,
                 snapshot_mode='gap_and_last', snapshot_gap=10, )
    experiment(variant)


    # train_tasks = [{'object': 0}, {'object': 1}]
    # eval_tasks = [{'object': 0}, {'object': 1}]
    # policy = TanhGaussianPolicy(
    #     obs_dim=cnn_params['output_size'],
    #     action_dim=action_dim,
    #     hidden_sizes=[256, 256, 256],
    #     obs_processor=policy_obs_processor,
    # )
    # algo_kwargs = {
    #     'num_iterations': 5,
    #     'meta_batch': 4,
    #     'embedding_batch_size': 256,
    #     'num_initial_steps': 2000,
    #     'num_steps_prior': 400,
    #     'num_steps_posterior': 0,
    #     'num_extra_rl_steps_posterior': 600,
    #     'num_train_steps_per_itr': 4000,
    #     'num_evals': 10, # number of independent evals per task
    #     'num_exp_traj_eval': 2,
    #     'num_steps_per_eval': 1000, # number of steps to eval for
    # }
    # algorithm = MetaRLAlgorithm(
    #     agent=agent,
    #     env=expl_env,
    #     trainer=trainer,
    #     train_task_indices=train_task_indices,
    #     eval_task_indices=eval_task_indices,
    #     train_tasks=train_tasks,
    #     eval_tasks=eval_tasks,
    #     use_next_obs_in_context=variant['use_next_obs_in_context'],
    #     # env_info_sizes=get_env_info_sizes(expl_env),
    #     env_info_sizes={},  # TODO(avi) check what this is used for
    #     **algo_kwargs
    # )
    # load_buffer_kwargs = {
    #     'pretrain_buffer_path': "21-02-22-ant-awac--exp7-ant-dir-4-eval-4-train-"
    #                             "sac-to-get-buffer-longer/21-02-22-ant-awac--exp7"
    #                             "-ant-dir-4-eval-4-train-sac-to-get-buffer-longer"
    #                             "_2021_02_23_06_09_23_id000--s270987/extra_snapshot_itr400.cpkl"
    # }
    # saved_tasks_path = "demos/ant_four_dir/buffer_550k_each/tasks.pkl"
    # load_buffer_kwargs = {
    #     'pretrain_buffer_path': ''
    # }
    # load_buffer_onto_algo(
    #     algorithm.replay_buffer,
    #     algorithm.enc_replay_buffer,
    #     **load_buffer_kwargs)