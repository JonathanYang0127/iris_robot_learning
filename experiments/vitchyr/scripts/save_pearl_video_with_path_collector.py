import argparse
import json
from pathlib import Path

import cloudpickle
import numpy as np
import torch

import rlkit.torch.pytorch_util as ptu
from multiworld.envs.pygame.point2d import Point2DEnv
from rlkit.core.meta_rl_algorithm import MetaRLAlgorithm
from rlkit.data_management.multitask_replay_buffer import MultiTaskReplayBuffer
from rlkit.envs.images import GymEnvRenderer, EnvRenderer
# from rlkit.envs.images.text_renderer import TextRenderer
from rlkit.envs.images.plot_renderer import TextRenderer, ScrollingPlotRenderer
from rlkit.envs.pearl_envs import HalfCheetahDirEnv, AntDirEnv, PointEnv
from rlkit.envs.wrappers.flat_to_dict import FlatToDictEnv
from rlkit.misc.asset_loader import load_local_or_remote_file
from rlkit.torch.pearl import video
from rlkit.torch.pearl.agent import MakePEARLAgentDeterministic
from rlkit.torch.pearl.buffer import PearlReplayBuffer
from rlkit.torch.pearl.diagnostics import (
    DebugInsertImagesEnv,
    FlatToDictPearlPolicy,
)
from rlkit.torch.pearl.launcher_util import load_buffer_onto_algo
from rlkit.torch.pearl.path_collector import (
    PearlPathCollector,
    PearlJointPathCollector,
)
from rlkit.torch.pearl.sampler import rollout, rollout_multiple_and_flatten
from rlkit.visualization.video import dump_video
from rlkit.launchers.launcher_util import load_pyhocon_configs
import rlkit.pythonplusplus as ppp

counter = 0

def simulate_policy(args):
    if args.pause:
        import ipdb; ipdb.set_trace()
    path = Path(args.file)
    if path.is_dir():
        snapshot_dir = path
        extra_snapshot_path = snapshot_dir / 'extra_snapshot.cpkl'
    else:
        extra_snapshot_path = path
        snapshot_dir = extra_snapshot_path.parent
    buffer_data = cloudpickle.load(
        open(extra_snapshot_path, 'rb'))
    algorithm = buffer_data['algorithm']
    ptu.set_gpu_mode(True)
    algorithm.trainer.context_decoder = algorithm.trainer.context_encoder # hack to support to() call with versions that didn't have context decoder
    algorithm.to(ptu.device)
    expl_path_collector = algorithm.expl_data_collector
    eval_collector = algorithm.expl_data_collector
    train_task_indices = algorithm.train_task_indices
    if isinstance(algorithm, MetaRLAlgorithm):
        test_task_indices = algorithm.eval_task_indices
        env = algorithm.env
        # import ipdb; ipdb.set_trace()
        # variant_path = snapshot_dir / 'variant.json'
        # variant = json.load(open(variant_path, 'rb'))
        pearl_replay_buffer = PearlReplayBuffer(
            algorithm.replay_buffer,
            algorithm.enc_replay_buffer,
            train_task_indices=train_task_indices,
            embedding_batch_size=algorithm.embedding_batch_size,
            meta_batch_size=algorithm.meta_batch,
        )
    else:
        test_task_indices = algorithm.test_task_indices
        env = algorithm.expl_env
        pearl_replay_buffer = algorithm.replay_buffer
    configs = [
        Path(__file__).parent.parent / 'new_pearl/configs/default_base.conf'''
    ]
    default_variant = ppp.recursive_to_dict(load_pyhocon_configs(configs))
    name_to_eval_path_collector_kwargs = default_variant['name_to_eval_path_collector_kwargs']
    name_to_expl_path_collector_kwargs = default_variant['name_to_expl_path_collector_kwargs']

    def create_eval_path_collector(env, policy):
        eval_path_collectors = {
            'train/' + name: PearlPathCollector(
                env, policy, train_task_indices, pearl_replay_buffer, **kwargs)
            for name, kwargs in name_to_eval_path_collector_kwargs.items()
        }
        eval_path_collectors.update({
            'test/' + name: PearlPathCollector(
                env, policy, test_task_indices,
                pearl_replay_buffer,
                **kwargs)
            for name, kwargs in name_to_eval_path_collector_kwargs.items()
        })
        return PearlJointPathCollector(eval_path_collectors)

    def create_expl_path_collector(env, policy):
        return PearlJointPathCollector({
            name: PearlPathCollector(
                env, policy, train_task_indices,
                pearl_replay_buffer,
                **kwargs)
            for name, kwargs in name_to_expl_path_collector_kwargs.items()
        })
    save_video_kwargs = dict(
        save_video_period=25,
        video_img_size=128,
        logdir=snapshot_dir / 'generated_rollouts'
    )

    n_eval_video_rollouts = (
            len(name_to_eval_path_collector_kwargs)
            * (len(train_task_indices) + len(test_task_indices)))
    eval_save_video_fn = video.make_save_video_function(
        env=env,
        policy=algorithm.trainer.agent,
        tag='retroactive_eval',
        create_path_collector=create_eval_path_collector,
        num_steps=n_eval_video_rollouts * algorithm.max_path_length,
        task_indices=train_task_indices,
        max_path_length=algorithm.max_path_length,
        discard_incomplete_paths=False,
        **save_video_kwargs)
    eval_save_video_fn(algorithm, 0)

    n_expl_video_rollouts = (
            len(name_to_expl_path_collector_kwargs)
            * len(train_task_indices))
    expl_save_video_fn = video.make_save_video_function(
        env=env,
        policy=algorithm.trainer.agent,
        tag='retroactive_expl',
        create_path_collector=create_expl_path_collector,
        num_steps=n_expl_video_rollouts * algorithm.max_path_length,
        task_indices=train_task_indices,
        max_path_length=algorithm.max_path_length,
        discard_incomplete_paths=False,
        **save_video_kwargs)
    expl_save_video_fn(algorithm, 0)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=300,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=10,
                        help='Speedup')
    parser.add_argument('--mode', default='video_env', type=str,
                        help='env mode')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--pause', action='store_true')
    parser.add_argument('--enable_render', action='store_true')
    parser.add_argument('--multitaskpause', action='store_true')
    parser.add_argument('--hide', action='store_true')
    args = parser.parse_args()

    simulate_policy(args)
