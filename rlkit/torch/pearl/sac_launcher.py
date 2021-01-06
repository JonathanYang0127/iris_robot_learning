import gym
# import roboverse
from rlkit.core.meta_rl_algorithm import MetaRLAlgorithm
from rlkit.data_management.awr_env_replay_buffer import AWREnvReplayBuffer
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.data_management.split_buffer import SplitReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv, StackObservationEnv, RewardWrapperEnv
import rlkit.torch.pytorch_util as ptu
from rlkit.samplers.data_collector import MdpPathCollector, ObsDictPathCollector
from rlkit.samplers.data_collector.step_collector import MdpStepCollector
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.pearl.path_collector import PearlPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.torch_rl_algorithm import (
    TorchBatchRLAlgorithm,
)
from rlkit.torch.pearl.agent import PEARLAgent
from rlkit.torch.pearl.pearl_trainer import PEARLSoftActorCriticTrainer
from rlkit.envs.pearl_envs import ENVS, register_pearl_envs

from rlkit.demos.source.hdf5_path_loader import HDF5PathLoader
from rlkit.demos.source.mdp_path_loader import MDPPathLoader
from rlkit.visualization.video import save_paths, VideoSaveFunction

from multiworld.core.flat_goal_env import FlatGoalEnv
from multiworld.core.image_env import ImageEnv
from multiworld.core.gym_to_multi_env import GymToMultiEnv
from rlkit.misc.hyperparameter import recursive_dictionary_update
from rlkit.torch.pearl.encoder import MlpEncoder, RecurrentEncoder

import torch
import numpy as np
from torchvision.utils import save_image

from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.gaussian_and_epislon import GaussianAndEpislonStrategy
from rlkit.exploration_strategies.ou_strategy import OUStrategy

import os.path as osp
from rlkit.core import logger
from rlkit.misc.asset_loader import load_local_or_remote_file
import pickle

from rlkit.envs.images import Renderer, InsertImageEnv, EnvRenderer
from rlkit.envs.make_env import make

ENV_PARAMS = {
    'HalfCheetah-v2': {
        'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 1000,
        'env_demo_path': dict(
            path="demos/icml2020/mujoco/hc_action_noise_15.npy",
            obs_dict=False,
            is_demo=True,
        ),
        'env_offpolicy_data_path': dict(
            path="demos/icml2020/mujoco/hc_off_policy_15_demos_100.npy",
            obs_dict=False,
            is_demo=False,
            train_split=0.9,
        ),
    },
    'Ant-v2': {
        'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 1000,
        'env_demo_path': dict(
            path="demos/icml2020/mujoco/ant_action_noise_15.npy",
            obs_dict=False,
            is_demo=True,
        ),
        'env_offpolicy_data_path': dict(
            path="demos/icml2020/mujoco/ant_off_policy_15_demos_100.npy",
            obs_dict=False,
            is_demo=False,
            train_split=0.9,
        ),
    },
    'Walker2d-v2': {
        'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 1000,
        'env_demo_path': dict(
            path="demos/icml2020/mujoco/walker_action_noise_15.npy",
            obs_dict=False,
            is_demo=True,
        ),
        'env_offpolicy_data_path': dict(
            path="demos/icml2020/mujoco/walker_off_policy_15_demos_100.npy",
            obs_dict=False,
            is_demo=False,
            train_split=0.9,
        ),
    },

    'SawyerRigGrasp-v0': {
        'env_id': 'SawyerRigGrasp-v0',
        # 'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 50,
        # 'num_epochs': 1000,
    },

    'pen-binary-v0': {
        'env_id': 'pen-binary-v0',
        'max_path_length': 200,
        'sparse_reward': True,
        'env_demo_path': dict(
            path="demos/icml2020/hand/pen2_sparse.npy",
            # path="demos/icml2020/hand/sparsity/railrl_pen-binary-v0_demos.npy",
            obs_dict=True,
            is_demo=True,
        ),
        'env_offpolicy_data_path': dict(
            # path="demos/icml2020/hand/pen_bc_sparse1.npy",
            # path="demos/icml2020/hand/pen_bc_sparse2.npy",
            # path="demos/icml2020/hand/pen_bc_sparse3.npy",
            # path="demos/icml2020/hand/pen_bc_sparse4.npy",
            path="demos/icml2020/hand/pen_bc_sparse4.npy",
            # path="ashvin/icml2020/hand/sparsity/bc/pen-binary1/run10/id*/video_*_*.p",
            # sync_dir="ashvin/icml2020/hand/sparsity/bc/pen-binary1/run10",
            obs_dict=False,
            is_demo=False,
            train_split=0.9,
        ),
    },
    'door-binary-v0': {
        'env_id': 'door-binary-v0',
        'max_path_length': 200,
        'sparse_reward': True,
        'env_demo_path': dict(
            path="demos/icml2020/hand/door2_sparse.npy",
            # path="demos/icml2020/hand/sparsity/railrl_door-binary-v0_demos.npy",
            obs_dict=True,
            is_demo=True,
        ),
        'env_offpolicy_data_path': dict(
            # path="demos/icml2020/hand/door_bc_sparse1.npy",
            # path="demos/icml2020/hand/door_bc_sparse3.npy",
            path="demos/icml2020/hand/door_bc_sparse4.npy",
            # path="ashvin/icml2020/hand/sparsity/bc/door-binary1/run10/id*/video_*_*.p",
            # sync_dir="ashvin/icml2020/hand/sparsity/bc/door-binary1/run10",
            obs_dict=False,
            is_demo=False,
            train_split=0.9,
        ),
    },
    'relocate-binary-v0': {
        'env_id': 'relocate-binary-v0',
        'max_path_length': 200,
        'sparse_reward': True,
        'env_demo_path': dict(
            path="demos/icml2020/hand/relocate2_sparse.npy",
            # path="demos/icml2020/hand/sparsity/railrl_relocate-binary-v0_demos.npy",
            obs_dict=True,
            is_demo=True,
        ),
        'env_offpolicy_data_path': dict(
            # path="demos/icml2020/hand/relocate_bc_sparse1.npy",
            path="demos/icml2020/hand/relocate_bc_sparse4.npy",
            # path="ashvin/icml2020/hand/sparsity/bc/relocate-binary1/run10/id*/video_*_*.p",
            # sync_dir="ashvin/icml2020/hand/sparsity/bc/relocate-binary1/run10",
            obs_dict=False,
            is_demo=False,
            train_split=0.9,
        ),
    },
}


def resume(variant):
    data = load_local_or_remote_file(variant.get("pretrained_algorithm_path"), map_location="cuda")
    algo = data['algorithm']

    algo.num_epochs = variant['num_epochs']

    post_pretrain_hyperparams = variant["trainer_kwargs"].get("post_pretrain_hyperparams", {})
    algo.trainer.set_algorithm_weights(**post_pretrain_hyperparams)

    algo.train()

def process_args(variant):
    if env_id:
        env_params = ENV_PARAMS.get(env_id, {})
        recursive_dictionary_update(variant, env_params)

def pearl_experiment(
        qf_kwargs=None,
        vf_kwargs=None,
        trainer_kwargs=None,
        algo_kwargs=None,
        context_encoder_kwargs=None,
        policy_class=None,
        policy_kwargs=None,
        policy_path=None,
        normalize_env=True,
        env_name=None,
        env_id=None,
        env_class=None,
        env_kwargs=None,
        env_params=None,
        add_env_demos=False,
        path_loader_kwargs=None,
        env_demo_path=None,
        env_offpolicy_data_path=None,
        add_env_offpolicy_data=False,
        exploration_kwargs=None,
        replay_buffer_class=EnvReplayBuffer,
        replay_buffer_kwargs=None,
        use_validation_buffer=False,
        pretrain_policy=False,
        pretrain_rl=False,
        train_rl=False,
        path_loader_class=MDPPathLoader,
        latent_dim=None,
        # video/debug
        debug=False,
        save_video=False,
        presampled_goals=None,
        renderer_kwargs=None,
        image_env_kwargs=None,
        save_paths=False,
        load_demos=False,
        load_env_dataset_demos=False,
        save_initial_buffers=False,
        save_pretrained_algorithm=False,
        # PEARL
        n_train_tasks=0,
        n_eval_tasks=0,
        path_to_weights=None,
        util_params=None,
):
    register_pearl_envs()
    env_kwargs = env_kwargs or {}
    env_params = env_params or {}
    path_loader_kwargs = path_loader_kwargs or {}
    exploration_kwargs = exploration_kwargs or {}
    replay_buffer_kwargs = replay_buffer_kwargs or {}
    context_encoder_kwargs = context_encoder_kwargs or {}
    trainer_kwargs = trainer_kwargs or {}
    # expl_env = make(env_id, env_class, env_kwargs, normalize_env)
    # eval_env = make(env_id, env_class, env_kwargs, normalize_env)
    expl_env = NormalizedBoxEnv(ENVS[env_name](**env_params))
    eval_env = NormalizedBoxEnv(ENVS[env_name](**env_params))
    reward_dim = 1

    if debug:
        algo_kwargs['max_path_length'] = 50
        algo_kwargs['batch_size'] = 5
        algo_kwargs['num_epochs'] = 5
        algo_kwargs['num_eval_steps_per_epoch'] = 100
        algo_kwargs['num_expl_steps_per_train_loop'] = 100
        algo_kwargs['num_trains_per_train_loop'] = 10
        algo_kwargs['min_num_steps_before_training'] = 100

    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    if hasattr(expl_env, 'info_sizes'):
        env_info_sizes = expl_env.info_sizes
    else:
        env_info_sizes = dict()

    context_encoder_input_dim = obs_dim + action_dim + reward_dim
    context_encoder_output_dim = latent_dim * 2
    def create_qf():
        return ConcatMlp(
            input_size=obs_dim + action_dim + latent_dim,
            output_size=1,
            **qf_kwargs
        )

    qf1 = create_qf()
    qf2 = create_qf()
    target_qf1 = create_qf()
    target_qf2 = create_qf()
    vf = ConcatMlp(
        input_size=obs_dim + latent_dim,
        output_size=1,
        **vf_kwargs
    )
    # target_qf1 = create_qf()
    # target_qf2 = create_qf()

    policy = TanhGaussianPolicy(
        obs_dim=obs_dim + latent_dim,
        action_dim=action_dim,
        **policy_kwargs,
    )
    context_encoder = MlpEncoder(
        input_size=context_encoder_input_dim,
        output_size=context_encoder_output_dim,
        hidden_sizes=[200, 200, 200],
        **context_encoder_kwargs
    )
    reward_predictor = ConcatMlp(
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
        hidden_sizes=[200, 200, 200],
    )
    agent = PEARLAgent(
        latent_dim,
        context_encoder,
        policy,
        reward_predictor,
    )

    eval_policy = MakeDeterministic(policy)
    eval_path_collector = PearlPathCollector(eval_env, eval_policy)
    expl_policy = policy

    trainer = PEARLSoftActorCriticTrainer(
        latent_dim=latent_dim,
        agent=agent,
        qf1=qf1,
        qf2=qf2,
        vf=vf,
        reward_predictor=reward_predictor,
        context_encoder=context_encoder,
        # target_qf1=target_qf1,
        # target_qf2=target_qf2,
        **trainer_kwargs
    )
    expl_path_collector = PearlPathCollector(expl_env, expl_policy)
    agent = PEARLAgent(
        latent_dim,
        context_encoder,
        policy,
        reward_predictor,
    )
    tasks = expl_env.get_all_task_idx()
    algorithm = MetaRLAlgorithm(
        agent=agent,
        env=expl_env,
        trainer=trainer,
        # exploration_env=expl_env,
        # evaluation_env=eval_env,
        # exploration_data_collector=expl_path_collector,
        # evaluation_data_collector=eval_path_collector,
        train_tasks=list(tasks[:n_train_tasks]),
        eval_tasks=list(tasks[-n_eval_tasks:]),
        # nets=[agent, qf1, qf2, vf],
        # latent_dim=latent_dim,
        **algo_kwargs
    )

    # algorithm = TorchBatchRLAlgorithm(
    #     trainer=trainer,
    #     exploration_env=expl_env,
    #     evaluation_env=eval_env,
    #     exploration_data_collector=expl_path_collector,
    #     evaluation_data_collector=eval_path_collector,
    #     **algo_kwargs
    # )
    algorithm.to(ptu.device)

    algorithm.train()
