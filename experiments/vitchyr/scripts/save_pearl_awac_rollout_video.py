import argparse
import json
from pathlib import Path

import cloudpickle
import numpy as np
import torch

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.multitask_replay_buffer import MultiTaskReplayBuffer
from rlkit.envs.images import GymEnvRenderer
# from rlkit.envs.images.text_renderer import TextRenderer
from rlkit.envs.images.plot_renderer import TextRenderer, ScrollingPlotRenderer
from rlkit.envs.wrappers.flat_to_dict import FlatToDictEnv
from rlkit.misc.asset_loader import load_local_or_remote_file
from rlkit.torch.pearl.buffer import PearlReplayBuffer
from rlkit.torch.pearl.diagnostics import (
    DebugInsertImagesEnv,
    FlatToDictPearlPolicy,
)
from rlkit.torch.pearl.launcher_util import load_buffer_onto_algo
from rlkit.torch.pearl.sampler import rollout
from rlkit.visualization.video import dump_video


def simulate_policy(args):
    if args.pause:
        import ipdb; ipdb.set_trace()
    # buffer_path = '/home/vitchyr/mnt2/log2/21-02-16-new-pearl--tmp-hc-create-buffer/21-02-16-new-pearl--tmp-hc-create-buffer_2021_02_16_15_08_12_id000--s22161/extra_snapshot_itr0.cpkl'
    # buffer_data = cloudpickle.load(
    #     open(buffer_path, 'rb'))
    # replay_buffer = buffer_data['replay_buffer']

    snapshot_path = args.file
    ptu.set_gpu_mode(True)
    data = torch.load(snapshot_path, map_location=ptu.device)
    pearl_replay_buffer = None
    if 'evaluation/test/posterior_live_update/policy' in data:
        policy = data['evaluation/test/posterior_live_update/policy']
        env = data['evaluation/test/posterior_live_update/env']
        extra_snapshot_path = Path(snapshot_path).parent / 'extra_snapshot.cpkl'
        if extra_snapshot_path.exists():
            buffer_data = cloudpickle.load(
                open(extra_snapshot_path, 'rb'))
            if 'algorithm' in buffer_data:
                algorithm = buffer_data['algorithm']
                pearl_replay_buffer = algorithm.replay_buffer
            else:
                pearl_replay_buffer = buffer_data['replay_buffer']
        variant_path = Path(snapshot_path).parent / 'variant.json'
        variant = json.load(open(variant_path, 'rb'))
        load_buffer_kwargs = variant['load_buffer_kwargs']
        saved_tasks_path = variant['saved_tasks_path']
        task_data = load_local_or_remote_file(
            saved_tasks_path, file_type='joblib')
        tasks = task_data['tasks']
        env.wrapped_env.tasks = tasks
    else:  # old-style trainer
        policy = data['agent']
        env = data['env']

        variant_path = Path(snapshot_path).parent / 'variant.json'
        variant = json.load(open(variant_path, 'rb'))
        import ipdb; ipdb.set_trace()
        load_buffer_kwargs = variant['load_buffer_kwargs']
        saved_tasks_path = variant['saved_tasks_path']
        use_ground_truth_context = variant.get('use_ground_truth_context', False)

        task_data = load_local_or_remote_file(
            saved_tasks_path, file_type='joblib')
        tasks = task_data['tasks']
        train_task_indices = task_data['train_task_indices']
        test_task_indices = task_data['eval_task_indices']
        env.wrapped_env.tasks = tasks
        task_indices = list(env.wrapped_env.get_all_task_idx())
        unwrapped_tasks = [
            np.array([t['goal']]) for t in tasks
        ]
        replay_buffer_kwargs = dict(
            max_replay_buffer_size=1000000,
            use_next_obs_in_context=False,
            sparse_rewards=False,
            use_ground_truth_context=use_ground_truth_context,
            ground_truth_tasks=unwrapped_tasks,
        )
        replay_buffer = MultiTaskReplayBuffer(
            env=env,
            task_indices=task_indices,
            **replay_buffer_kwargs
        )
        enc_replay_buffer = MultiTaskReplayBuffer(
            env=env,
            task_indices=task_indices,
            **replay_buffer_kwargs
        )
        pearl_buffer_kwargs=dict(
            meta_batch_size=4,
            embedding_batch_size=256,
        )
        pearl_replay_buffer = PearlReplayBuffer(
            replay_buffer,
            enc_replay_buffer,
            train_task_indices=train_task_indices,
            **pearl_buffer_kwargs
        )
    load_buffer_onto_algo(
        pearl_replay_buffer.replay_buffer,
        pearl_replay_buffer.encoder_replay_buffer,
        **load_buffer_kwargs
    )

    obs_key = 'tmp'
    policy = FlatToDictPearlPolicy(policy, obs_key)
    env = FlatToDictEnv(env, obs_key)

    img_renderer = GymEnvRenderer(
        width=256,
        height=256,
    )
    text_renderer = TextRenderer(
        text='test',
        width=256,
        height=256,
    )
    reward_plotter = ScrollingPlotRenderer(
        width=256,
        height=256,
    )
    renderers={
        'image_observation': img_renderer,
        'text': text_renderer,
        'reward': reward_plotter,
    }
    img_env = DebugInsertImagesEnv(
        wrapped_env=env,
        renderers=renderers,
    )

    save_dir = Path(snapshot_path).parent / 'generated_rollouts'
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / 'rollout0.mp4'
    # rollout_fn = partial(
    #     rollout,
    # )

    def random_task_rollout_fn(*args, **kwargs):
        task_idx = np.random.choice([0, 1])
        if pearl_replay_buffer is not None:
            init_context = pearl_replay_buffer.sample_context(task_idx)
            init_context = ptu.from_numpy(init_context)
        else:
            init_context = None
        return rollout(
            *args,
            task_idx=task_idx,
            initial_context=init_context,
            resample_latent_period=1,
            # accum_context=True,
            # update_posterior_period=1,
            **kwargs)
    dump_video(
        env=img_env,
        policy=policy,
        filename=save_path,
        rollout_function=random_task_rollout_fn,
        obs_dict_key='observations',
        keys_to_show=list(renderers.keys()),
        image_format=img_renderer.output_image_format,
        # rows=1,
        # columns=1,
        rows=2,
        columns=3,
        imsize=256,
    )


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
