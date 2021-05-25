
import argparse
import time
import os.path as osp
import os

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from rlkit.torch.sac.awac_trainer import AWACTrainer
from rlkit.torch.sac.policies import GaussianCNNPolicy
from rlkit.torch.networks.cnn import CNN, ConcatCNN

from rlkit.data_management.obs_dict_replay_buffer import ObsDictReplayBuffer
from rlkit.samplers.data_collector import MdpPathCollector, ObsDictPathCollector
from rlkit.launchers.launcher_util import setup_logger
from rlkit.core import logger
from rlkit.torch.networks import Clamp
from rlkit.pythonplusplus import identity

import roboverse
import numpy as np

CUSTOM_LOG_DIR = '/nfs/kun1/users/avi/doodad-output/'
LOCAL_LOG_DIR = '/media/avi/data/Work/doodad_output/'

BUFFER = '/media/avi/data/Work/github/avisingh599/minibullet/data/may14_meta_Widow250MultiTaskGraspShed-v0_1000_save_all_noise_0.1_2021-05-14T16-27-16/may14_meta_Widow250MultiTaskGraspShed-v0_1000_save_all_noise_0.1_2021-05-14T16-27-16_1000.npy'


class VideoSaveFunctionBullet:
    def __init__(self, variant):
        self.logdir = logger.get_snapshot_dir()
        self.dump_video_kwargs = variant.get("dump_video_kwargs", dict())
        self.save_period = self.dump_video_kwargs.pop('save_video_period', 5)

    def __call__(self, algo, epoch):
        if epoch % self.save_period == 0 or epoch == algo.num_epochs:
            video_dir = osp.join(self.logdir,
                                 'videos_eval/{epoch}/'.format(epoch=epoch))
            eval_paths = algo.eval_data_collector.get_epoch_paths()
            dump_video_basic(video_dir, eval_paths)


def dump_video_basic(video_dir, paths):
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)

    import cv2
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    for i, path in enumerate(paths):
        video = path['next_observations']
        frame_list = []
        for frame in video:
            # TODO(avi) Figure out why this hack is needed
            # if isinstance(frame, np.ndarray):
            #     if 'real_image' in frame[0]:
            #         frame_list.append(frame[0]['real_image'])
            #     else:
            #         frame_list.append(frame[0]['image'])
            # else:
            #     if 'real_image' in frame:
            #         frame_list.append(frame['real_image'])
            #     else:
            #         frame_list.append(frame['image'])

            frame_list.append(frame['image'])
        frame_list = np.asarray(frame_list)
        video_len = frame_list.shape[0]
        n_channels = 3
        imsize = int(np.sqrt(frame_list.shape[1] / n_channels))
        assert imsize*imsize*n_channels == frame_list.shape[1]

        video = frame_list.reshape(video_len, n_channels, imsize, imsize)
        video = np.transpose(video, (0, 2, 3, 1))
        video = (video*255.0).astype(np.uint8)
        filename = osp.join(video_dir, '{}.mp4'.format(i))
        FPS = float(np.ceil(video_len/3.0))
        writer = cv2.VideoWriter(filename, fourcc, FPS, (imsize, imsize))
        for j in range(video.shape[0]):
            writer.write(cv2.cvtColor(video[j], cv2.COLOR_RGB2BGR))
        writer = None


def get_buffer_size(data):
    num_transitions = 0
    for i in range(len(data)):
        for j in range(len(data[i]['observations'])):
            num_transitions += 1
    return num_transitions


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


def add_data_to_buffer(data, replay_buffer, observation_keys):

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
        replay_buffer.add_path(path)


def experiment(variant):
    eval_env = roboverse.make(variant['env'], transpose_image=True)
    expl_env = eval_env
    action_dim = eval_env.action_space.low.size
    observation_keys = ['image']

    cnn_params = variant['cnn_params']
    cnn_params.update(
        # output_size=action_dim,
        added_fc_input_size=0,
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

    state_observation_dim = 0
    cnn_params.update(
        output_size=1,
        added_fc_input_size=state_observation_dim + action_dim,
    )

    if variant['use_negative_rewards']:
        cnn_params.update(output_activation=Clamp(max=0))  # rewards are <= 0

    qf1 = ConcatCNN(**cnn_params)
    qf2 = ConcatCNN(**cnn_params)
    target_qf1 = ConcatCNN(**cnn_params)
    target_qf2 = ConcatCNN(**cnn_params)

    with open(variant['buffer'], 'rb') as fl:
        data = np.load(fl, allow_pickle=True)
    num_transitions = get_buffer_size(data)
    max_replay_buffer_size = num_transitions + 10
    replay_buffer = ObsDictReplayBuffer(
        max_replay_buffer_size,
        expl_env,
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
    expl_path_collector = ObsDictPathCollector(
        expl_env,
        policy,
        observation_key='image',
    )
    eval_path_collector = ObsDictPathCollector(
        expl_env,
        policy,
        observation_key='image',
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
    parser.add_argument('--use-negative-rewards', action='store_true',
                        default=False)

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
    if osp.isdir(CUSTOM_LOG_DIR):
        base_log_dir = CUSTOM_LOG_DIR
    else:
        base_log_dir = LOCAL_LOG_DIR
    setup_logger(logger, exp_prefix, base_log_dir, variant=variant,
                 snapshot_mode='gap_and_last', snapshot_gap=10, )

    experiment(variant)
