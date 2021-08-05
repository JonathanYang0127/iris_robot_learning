import argparse
import time
import os
import gym
from roboverse.bullet.serializable import Serializable

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from rlkit.torch.sac.awac_trainer import AWACTrainer
from rlkit.torch.sac.policies import GaussianCNNPolicy, MakeDeterministic
from rlkit.torch.networks.cnn import ConcatCNN
from rlkit.torch.sac.awac_joint_embedding_trainer import \
    AWACJointEmbeddingMultitaskTrainer

from rlkit.data_management.obs_dict_replay_buffer import ObsDictReplayBuffer
from rlkit.data_management.multitask_replay_buffer import \
    ObsDictMultiTaskReplayBuffer
from rlkit.samplers.data_collector import ObsDictPathCollector
from rlkit.launchers.launcher_util import setup_logger
from rlkit.core import logger
from rlkit.torch.networks import Clamp
from rlkit.misc.roboverse_utils import get_buffer_size_multitask, \
    add_data_to_buffer_multitask_v2, VideoSaveFunctionBullet

import roboverse
import numpy as np
from gym import spaces
from rlkit.launchers.config import LOCAL_LOG_DIR
from rlkit.torch.task_encoders.encoder_decoder_nets import EncoderDecoderNet, \
    VanillaEncoderNet

BUFFER = '/media/avi/data/sim_data/aug3_Widow250PickPlaceMetaTrainMultiObjectMultiContainer-v0_4K_save_all_noise_0.1_2021-08-03T15-06-13_3840.npy'


class EmbeddingWrapper(gym.Env, Serializable):

    def __init__(self, env, embedding_network, positive_buffer=None):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.embedding_network = embedding_network
        self.positive_buffer = positive_buffer

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        new_state_obs = np.concatenate([obs['state'], self.curr_embedding],
                                       axis=0)
        obs.update({'state': new_state_obs})
        return obs, rew, done, info

    def reset(self):
        obs = self.env.reset()
        positive_data = self.positive_buffer.sample_batch(
            [self.env.task_idx], 1,
        )
        goal_image = ptu.from_numpy(positive_data['observations'][0])
        self.curr_embedding = ptu.get_numpy(self.embedding_network(goal_image))[
            0]
        # a bit of a hack, we are just adding the embedding to the state
        new_state_obs = np.concatenate([obs['state'], self.curr_embedding],
                                       axis=0)
        obs.update({'state': new_state_obs})
        return obs

    def reset_task(self, task_idx):
        self.env.reset_task(task_idx)


def experiment(variant):
    num_tasks = variant['num_tasks']
    eval_env = roboverse.make(variant['env'], transpose_image=True,
                              num_tasks=num_tasks)
    action_dim = eval_env.action_space.low.size
    image_size = 48

    with open(variant['buffer'], 'rb') as fl:
        data = np.load(fl, allow_pickle=True)
    num_transitions = get_buffer_size_multitask(data)
    max_replay_buffer_size = num_transitions + 10

    if variant['use_robot_state']:
        observation_keys = ['image', 'state']
        state_observation_dim = eval_env.observation_space.spaces[
            'state'].low.size
    else:
        observation_keys = ['image', ]
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

    buffer_policy = GaussianCNNPolicy(max_log_std=0,
                                      min_log_std=-6,
                                      obs_dim=None,
                                      action_dim=action_dim,
                                      std_architecture="values",
                                      **cnn_params)

    cnn_params.update(
        output_size=1,
        added_fc_input_size=state_observation_dim + variant[
            'latent_dim'] + action_dim,
    )

    # if variant['use_negative_rewards']:
    #     cnn_params.update(output_activation=Clamp(max=0))  # rewards are <= 0
    concat_cnn_class = ConcatCNN
    qf1 = concat_cnn_class(**cnn_params)
    qf2 = concat_cnn_class(**cnn_params)
    target_qf1 = concat_cnn_class(**cnn_params)
    target_qf2 = concat_cnn_class(**cnn_params)

    task_encoder = VanillaEncoderNet(variant['latent_dim'], image_size,
                                     image_augmentation=variant[
                                         'encoder_image_aug'])

    eval_env = EmbeddingWrapper(eval_env, task_encoder)
    expl_env = eval_env

    replay_buffer = ObsDictMultiTaskReplayBuffer(
        max_replay_buffer_size,
        expl_env,
        np.arange(num_tasks),
        use_next_obs_in_context=variant['use_next_obs_in_context'],
        sparse_rewards=False,
        observation_keys=observation_keys
    )
    add_data_to_buffer_multitask_v2(data, replay_buffer, observation_keys)

    # add positive images to the positive buffer
    positive_sizes = []
    for i in range(num_tasks):
        buffer = replay_buffer.task_buffers[i]
        counter = 0
        for j in range(buffer._top):
            if buffer._rewards[j]:
                counter += 1
        positive_sizes.append(counter)

    max_positive_buffer_size = np.max(positive_sizes) + 10

    # initialize positive buffer, add data
    replay_buffer_positive = ObsDictMultiTaskReplayBuffer(
        max_positive_buffer_size,
        expl_env,
        np.arange(num_tasks),
        use_next_obs_in_context=False,
        sparse_rewards=False,
        observation_keys=observation_keys
    )
    for i in range(num_tasks):
        buffer = replay_buffer.task_buffers[i]
        for j in range(buffer._top):
            if buffer._rewards[j]:
                obs_dict = {}
                next_obs_dict = {}
                for key in observation_keys:
                    obs_dict[key] = buffer._obs[key][j]
                    next_obs_dict[key] = buffer._next_obs[key][j]
                replay_buffer_positive.add_sample(
                    i, obs_dict, buffer._actions[j], buffer._rewards[j],
                    buffer._terminals[j], next_obs_dict,
                )

    trainer = AWACJointEmbeddingMultitaskTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        task_encoder=task_encoder,
        buffer_policy=buffer_policy,
        **variant['trainer_kwargs']
    )

    eval_env.positive_buffer = replay_buffer_positive
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
        train_embedding_network=True,
        replay_buffer_positive=replay_buffer_positive,
        max_path_length=variant['max_path_length'],
        batch_size=variant['batch_size'],
        multi_task=True,
        train_tasks=np.arange(num_tasks),
        eval_tasks=np.arange(num_tasks),
        meta_batch_size=variant['meta_batch_size'],
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

    # import IPython; IPython.embed()


def enable_gpus(gpu_str):
    if gpu_str != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str,
                        default='Widow250PickPlaceMetaTrainMultiObjectMultiContainer-v0')
    parser.add_argument("--num-tasks", type=int, default=8)
    parser.add_argument("--buffer", type=str, default=BUFFER)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--gpu", default='0', type=str)
    args = parser.parse_args()

    variant = dict(
        algorithm='AWAC-JointEmbedding',
        num_epochs=3000,
        batch_size=64,
        meta_batch_size=4,
        max_path_length=30,
        num_trains_per_train_loop=1000,
        # num_eval_steps_per_epoch=0,
        num_eval_steps_per_epoch=120,
        num_expl_steps_per_train_loop=0,
        min_num_steps_before_training=0,

        # for task encoder
        latent_dim=2,
        encoder_image_aug=True,
        use_next_obs_in_context=False,

        dump_video_kwargs=dict(
            save_video_period=10,
        ),

        env=args.env,
        num_tasks=args.num_tasks,
        buffer=args.buffer,
        # use_negative_rewards=args.use_negative_rewards,
        use_robot_state=True,
        # use_task_embedding=args.use_task_embedding,
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

    enable_gpus(args.gpu)
    ptu.set_gpu_mode(True)

    exp_prefix = '{}-awac-image-{}'.format(time.strftime("%y-%m-%d"), args.env)
    setup_logger(logger, exp_prefix, LOCAL_LOG_DIR, variant=variant,
                 snapshot_mode='gap_and_last', snapshot_gap=10, )

    experiment(variant)
