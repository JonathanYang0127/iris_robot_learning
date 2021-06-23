import numpy as np
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.obs_dict_replay_buffer import \
    ObsDictReplayBuffer
from rlkit.samplers.data_collector import MdpPathCollector, ObsDictPathCollector
from rlkit.misc.wx250_utils import add_data_to_buffer_real_robot, DummyEnv

from rlkit.torch.networks import CNN
from rlkit.torch.sac.policies import ObservationConditionedRealNVP

from rlkit.torch.sac.rnvp_trainer import RealNVPTrainer
from rlkit.misc.roboverse_utils import add_data_to_buffer

from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from rlkit.launchers.config import LOCAL_LOG_DIR
from rlkit.launchers.launcher_util import setup_logger
from rlkit.core import logger

import time

DEFAULT_BUFFER = '/media/avi/data/Work/github/avisingh599/minibullet/data/june15_test_Widow250PickPlaceMedium-v0_100_noise_0.1_2021-06-15T14-36-15/june15_test_Widow250PickPlaceMedium-v0_100_noise_0.1_2021-06-15T14-36-15_100.npy'


def experiment(variant):

    if variant['env'] == 'robot':
        image_size = 64
        eval_env = DummyEnv(image_size=image_size, use_wrist=False)
    else:
        import roboverse
        eval_env = roboverse.make(variant['env'], transpose_image=True)
        image_size = 48

    expl_env = eval_env
    action_dim = eval_env.action_space.low.size
    img_width, img_height = image_size, image_size
    num_channels = 3

    if variant['use_robot_state']:
        observation_keys = ['image', 'state']
        state_observation_dim = eval_env.observation_space.spaces['state'].low.size
    else:
        observation_keys = ['image']
        state_observation_dim = 0

    cnn_params = variant['cnn_params']
    cnn_params.update(
        input_width=img_width,
        input_height=img_height,
        input_channels=num_channels,
    )
    if variant['use_robot_state']:
        robot_state_obs_dim = eval_env.observation_space.spaces['state'].low.size
        cnn_params.update(
            added_fc_input_size=robot_state_obs_dim,
            output_conv_channels=False,
            hidden_sizes=[400, 400, 200],
            # cnn_input_key=expl_env.cnn_input_key,
            # fc_input_key=expl_env.fc_input_key,
        )
    else:
        cnn_params.update(
            added_fc_input_size=0,
            # cnn_input_key=expl_env.cnn_input_key,
            output_conv_channels=False,
        )
    cnn = CNN(**cnn_params)

    assert variant['coupling_layers'] % 2 == 0
    flips = [False]
    for _ in range(variant['coupling_layers'] - 1):
        flips.append(not flips[-1])
    print('flips', flips)

    real_nvp_policy = ObservationConditionedRealNVP(
        flips,
        action_dim,
        obs_processor=cnn,
        ignore_observation=(not variant['observation_conditioning']),
        use_atanh_preprocessing=variant['use_atanh'],
    )

    expl_path_collector = ObsDictPathCollector(
        expl_env,
        real_nvp_policy,
        observation_keys=observation_keys,
    )
    eval_path_collector = ObsDictPathCollector(
        eval_env,
        real_nvp_policy,
        observation_keys=observation_keys,
    )

    replay_buffer = ObsDictReplayBuffer(
        int(1E6),
        expl_env,
        observation_keys=observation_keys
    )

    if variant['env'] == 'robot':
        add_data_to_buffer_real_robot(variant['buffer'], replay_buffer,
                                      validation_replay_buffer=None,
                                      validation_fraction=0.8, num_trajs_limit=variant['num_trajs_limit'])
    else:
        with open(variant['buffer'], 'rb') as fl:
            data = np.load(fl, allow_pickle=True)
        add_data_to_buffer(data, replay_buffer, observation_keys)

    trainer = RealNVPTrainer(
        env=eval_env,
        bijector=real_nvp_policy,
        **variant['trainer_kwargs']
    )

    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algo_kwargs']
    )

    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    variant = dict(
        trainer_kwargs=dict(
            lr=1e-4,
        ),
        algo_kwargs=dict(
            batch_size=256,
            max_path_length=30,
            num_epochs=2000,
            num_eval_steps_per_epoch=0,
            num_expl_steps_per_train_loop=0,
            num_trains_per_train_loop=1000,
            min_num_steps_before_training=0,
            # max_path_length=10,
            # num_epochs=100,
            # num_eval_steps_per_epoch=100,
            # num_expl_steps_per_train_loop=100,
            # num_trains_per_train_loop=100,
            # min_num_steps_before_training=100,
        ),
        cnn_params=dict(
            kernel_sizes=[3, 3],
            n_channels=[4, 4],
            strides=[1, 1],
            hidden_sizes=[200, 200],
            paddings=[1, 1],
            pool_type='max2d',
            pool_sizes=[2, 2],
            pool_strides=[2, 2],
            pool_paddings=[0, 0],
            output_size=32,
            image_augmentation=False,
        ),
        policy_kwargs=dict(
            hidden_sizes=[256, 256],
        ),
        dump_video_kwargs=dict(
            imsize=48,
            save_video_period=1,
        ),
        logger_config=dict(
            snapshot_mode='gap_and_last',
            snapshot_gap=10,
        ),
        dump_buffer_kwargs=dict(
            dump_buffer_period=50,
        ),
        replay_buffer_size=int(5E5),
        expl_path_collector_kwargs=dict(),
        eval_path_collector_kwargs=dict(),
        shared_qf_conv=False,
        use_robot_state=False,
        randomize_env=True,
        batch_rl=True,
    )

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--buffer", type=str, default=DEFAULT_BUFFER)
    parser.add_argument("--buffer-val", type=str, default='')
    parser.add_argument("--obs", default='pixels',
                        type=str, choices=('pixels', 'pixels_debug'))
    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument("--no-obs-input", action="store_true", default=False)
    parser.add_argument("--eval-test-objects", action="store_true", default=False)
    parser.add_argument("--use-robot-state", action="store_true", default=False)
    parser.add_argument("--cnn", type=str, default='large',
                        choices=('small', 'large', 'xlarge'))
    parser.add_argument("--coupling-layers", type=int, default=4)
    parser.add_argument("--cnn-output-size", type=int, default=256)
    parser.add_argument("--use-img-aug", action="store_true", default=False)
    parser.add_argument("--no-use-atanh", action="store_true", default=False)
    parser.add_argument("--use-grad-clip", action="store_true", default=False)
    parser.add_argument("--grad-clip-threshold", type=float, default=50.0)
    parser.add_argument("--num-trajs-limit", default=0, type=int)
    parser.add_argument('--seed', default=10, type=int)

    args = parser.parse_args()

    variant['env'] = args.env
    variant['obs'] = args.obs
    variant['buffer'] = args.buffer
    variant['buffer_validation'] = args.buffer_val

    variant['observation_conditioning'] = not args.no_obs_input
    variant['eval_test_objects'] = args.eval_test_objects
    variant['use_robot_state'] = args.use_robot_state
    variant['coupling_layers'] = args.coupling_layers

    variant['cnn'] = args.cnn

    if variant['cnn'] == 'small':
        pass
    elif variant['cnn'] == 'large':
        variant['cnn_params'].update(
            kernel_sizes=[3, 3, 3],
            n_channels=[16, 16, 16],
            strides=[1, 1, 1],
            hidden_sizes=[1024, 512],
            paddings=[1, 1, 1],
            pool_type='max2d',
            pool_sizes=[2, 2, 1], # the one at the end means no pool
            pool_strides=[2, 2, 1],
            pool_paddings=[0, 0, 0],
        )
    elif variant['cnn'] == 'xlarge':
        variant['cnn_params'].update(
            kernel_sizes=[3, 3, 3, 3],
            n_channels=[32, 32, 32, 32],
            strides=[1, 1, 1, 1],
            hidden_sizes=[1024, 512, 512],
            paddings=[1, 1, 1, 1],
            pool_type='max2d',
            pool_sizes=[2, 2, 1, 1], # the one at the end means no pool
            pool_strides=[2, 2, 1, 1],
            pool_paddings=[0, 0, 0, 0],
        )

    variant['cnn_params']['output_size'] = args.cnn_output_size
    variant['cnn_params']['image_augmentation'] = args.use_img_aug
    variant['use_atanh'] = not args.no_use_atanh
    variant['trainer_kwargs']['clip_gradients_by_norm'] = args.use_grad_clip
    variant['trainer_kwargs']['clip_gradients_by_norm_threshold'] = \
        args.grad_clip_threshold
    variant['seed'] = args.seed

    if args.num_trajs_limit > 0:
        variant['num_trajs_limit'] = args.num_trajs_limit
    else:
        variant['num_trajs_limit'] = None

    def enable_gpus(gpu_str):
        import os
        if gpu_str != "":
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
        return

    enable_gpus(args.gpu)
    ptu.set_gpu_mode(True)

    exp_prefix = '{}-rnvp-{}'.format(time.strftime("%y-%m-%d"), args.env)
    setup_logger(logger, exp_prefix, LOCAL_LOG_DIR, variant=variant,
                 snapshot_mode='gap_and_last', snapshot_gap=10, )
    experiment(variant)
