import argparse, os
import numpy as np

import h5py
import d4rl, gym

import rlkit.util.hyperparameter as hyp
from rlkit.launchers.arglauncher import run_variants

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector.path_collector import MdpPathCollector, CustomMDPPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic, PolicyFromQ
from rlkit.torch.sac.cql import CQLTrainer
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

def load_hdf5(dataset, replay_buffer):
    size = dataset['terminals'].shape[0]
    replay_buffer._observations[:size] = dataset['observations']
    replay_buffer._next_obs[:size] = dataset['next_observations']
    replay_buffer._actions[:size] = dataset['actions']
    # Center reward for Ant-Maze
    replay_buffer._rewards[:size] = np.expand_dims(dataset['rewards'], 1)
    replay_buffer._terminals[:size] = np.expand_dims(dataset['terminals'], 1)
    print ('Number of terminals on: ', replay_buffer._terminals.sum())
    replay_buffer._top = size
    replay_buffer._size = size

def experiment(variant):
    eval_env = gym.make(variant['env_name'])
    expl_env = eval_env

    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    M = variant['layer_size']
    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M, M],
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M, M],
    )
    target_qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M, M],
    )
    target_qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M, M],
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M, M],
    )
    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    # expl_path_collector = CustomMDPPathCollector(
        # eval_env,
    # )
    expl_path_collector = MdpPathCollector(
        expl_env,
        PolicyFromQ(qf1, policy, num_samples=10),
    )
    buffer_filename = None
    if variant['buffer_filename'] is not None:
        buffer_filename = variant['buffer_filename']

    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
    if variant['load_buffer'] and buffer_filename is not None:
        replay_buffer.load_buffer(buffer_filename)
    else:
        load_hdf5(d4rl.qlearning_dataset(eval_env), replay_buffer)

    trainer = CQLTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs']
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        offline_rl=True,
        # eval_both=True,
        # batch_rl=variant['load_buffer'],
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()

if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algorithm="CQL",
        version="normal",
        layer_size=256,
        replay_buffer_size=int(2E6),
        buffer_filename=None,
        load_buffer=True,
        env_name='Hopper-v2',
        sparse_reward=False,
        algorithm_kwargs=dict(
            start_epoch=-1500,
            num_epochs=1000,
            num_eval_steps_per_epoch=1000,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=1000,
            batch_size=256,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            policy_lr=1E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,

            # Target nets/ policy vs Q-function update
            policy_eval_start=40000,
            num_qs=2,

            # min Q
            temp=1.0,
            min_q_version=3,
            min_q_weight=1.0,

            # lagrange
            with_lagrange=True,   # Defaults to true
            lagrange_thresh=5.0,

            # extra params
            num_random=10,
            max_q_backup=False,
            deterministic_backup=False,
            reward_transform_kwargs=dict(m=4, b=-2),
        ),
        # launcher_config=dict(
        #     mode="local_docker",
        # ),

    )

    search_space = {
        'env_name': [
            "antmaze-large-diverse-v0", "antmaze-large-play-v0",
        ],
        'trainer_kwargs.lagrange_thresh': [5.0],
        # 'trainer_kwargs.with_lagrange': [False],
        'trainer_kwargs.min_q_weight': [5.0, ],
        'seedid': range(3),
        'algorithm_kwargs.start_epoch': [-1500],
    }

    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    variants = []
    for variant in sweeper.iterate_hyperparameters():
        variants.append(variant)

    run_variants(experiment, variants, )

