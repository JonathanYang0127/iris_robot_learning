#### RLKIT FROM BATCH_RL_PRIVATE!

import argparse, os
import numpy as np

import h5py
import gym

# import gym_mujoco
import d4rl

def load_hdf5(dataset, replay_buffer):
    all_obs = dataset['observations']
    all_act = dataset['actions']
    N = min(all_obs.shape[0], 2000000)
    _obs = all_obs[:N]
    _actions = all_act[:N]
    _next_obs = np.concatenate([all_obs[1:N,:], np.zeros_like(_obs[0])[np.newaxis,:]], axis=0)
    _rew = dataset['rewards'][:N]
    _done = dataset['terminals'][:N]

    replay_buffer._observations = _obs
    replay_buffer._next_obs = _next_obs
    replay_buffer._actions = _actions
    replay_buffer._rewards = _rew # np.expand_dims(_rew, 1)
    replay_buffer._terminals = _done #np.expand_dims(_done, 1)
    replay_buffer._size = N-1
    replay_buffer._top = replay_buffer._size

def experiment(variant):
    from gym.envs.mujoco import HalfCheetahEnv, HopperEnv, AntEnv, Walker2dEnv, HumanoidEnv
    # from metaworld.envs.mujoco.sawyer_xyz.sawyer_peg_insertion_side import SawyerPegInsertionSideEnv
    # from metaworld.envs.mujoco.sawyer_xyz.sawyer_stick_pull import SawyerStickPullEnv
    # from metaworld.envs.mujoco.sawyer_xyz.sawyer_hammer import SawyerHammerEnv
    import rlkit.torch.pytorch_util as ptu
    from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
    from rlkit.envs.wrappers import NormalizedBoxEnv
    from rlkit.launchers.launcher_util import setup_logger
    from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
    from rlkit.torch.sac.sac_minq import SACTrainer
    from rlkit.torch.networks import FlattenMlp
    from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
    from rlkit.samplers.data_collector import MdpPathCollector, CustomMDPPathCollector
    eval_env = gym.make(variant['env_name'])
    expl_env = eval_env

    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    M = variant['layer_size']
    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M, M],
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M, M],
    )
    target_qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M, M],
    )
    target_qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M, M],
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M, M],    # Making it easier to visualize
    )
    # behavior_policy = TanhGaussianPolicy(
    #     obs_dim=obs_dim,
    #     action_dim=action_dim,
    #     hidden_sizes=[M, M],
    # )
    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
        sparse_reward=False,
    )
    expl_path_collector = CustomMDPPathCollector(
        expl_env,
    )
    buffer_filename = None
    if variant['buffer_filename'] is not None:
        buffer_filename = variant['buffer_filename']

    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
        with_per=False,
    )
    if variant['load_buffer'] and buffer_filename is not None:
        replay_buffer.load_buffer(buffer_filename)
    else:
        load_hdf5(eval_env.unwrapped.get_dataset(), replay_buffer)

    trainer = SACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        behavior_policy=None,
        **variant['trainer_kwargs']
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        eval_both=True,
        batch_rl=variant['load_buffer'],
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()

def enable_gpus(gpu_str):
    if (gpu_str is not ""):
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
    return

if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algorithm="SAC",
        version="normal",
        layer_size=256,
        replay_buffer_size=int(2E6),
        buffer_filename=None, #halfcheetah_101000.pkl',
        load_buffer=None,
        env_name='Hopper-v2',
        sparse_reward=False,
        algorithm_kwargs=dict(
            num_epochs=3000,
            num_eval_steps_per_epoch=5000,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=1000,
            batch_size=256,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=1E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,

            # Target nets/ policy vs Q-function update
            use_target_nets=True,
            policy_eval_start=40000,    #30000,  (This is for offline)
            num_qs=2,

            # min Q
            with_min_q=True,
            new_min_q=True,
            hinge_bellman=False,
            temp=10.0,
            min_q_version=0,
            use_projected_grad=False,
            normalize_magnitudes=False,
            regress_constant=False,
            min_q_weight=1.0,
            data_subtract=False,

            # extra params
            num_random=10,
            max_q_backup=False,
            deterministic_backup=False,
        ),
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--load_buffer", action='store_true')
    parser.add_argument("--env", type=str, default='halfcheetah-random-v0')
    parser.add_argument("--gpu", default='0', type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--data_subtract", type=str, default="False")
    parser.add_argument("--max_q_backup", type=str, default="False")
    parser.add_argument("--deterministic_backup", type=str, default="False")
    parser.add_argument("--policy_eval_start", default=40000, type=int)
    parser.add_argument('--use_projected_grad', default='False', type=str)
    parser.add_argument('--min_q_weight', default=1.0, type=float)
    parser.add_argument('--policy_lr', default=1e-4, type=float)
    parser.add_argument('--min_q_version', default=0, type=int)
    parser.add_argument('--temp', default=1.0, type=float)

    args = parser.parse_args()
    enable_gpus(args.gpu)
    variant['trainer_kwargs']['use_projected_grad'] = (True if args.use_projected_grad == 'True' else False)
    variant['trainer_kwargs']['max_q_backup'] = (True if args.max_q_backup == 'True' else False)
    variant['trainer_kwargs']['deterministic_backup'] = (True if args.deterministic_backup == 'True' else False)
    variant['trainer_kwargs']['min_q_weight'] = args.min_q_weight
    variant['trainer_kwargs']['policy_lr'] = args.policy_lr
    variant['trainer_kwargs']['min_q_version'] = args.min_q_version
    variant['trainer_kwargs']['temp'] = args.temp
    variant['trainer_kwargs']['data_subtract'] = (True if args.data_subtract == "True" else False)
    variant['trainer_kwargs']['policy_eval_start'] = args.policy_eval_start
    variant['seed'] = args.seed

    variant['buffer_filename'] = None

    variant['load_buffer'] = True
    variant['env_name'] = args.env

    rnd = np.random.randint(0, 1000000)
    setup_logger(os.path.join('min_Q_sac_offline_with_IS', str(rnd)), variant=variant, base_log_dir='./data')
    ptu.set_gpu_mode(True)
    experiment(variant)
