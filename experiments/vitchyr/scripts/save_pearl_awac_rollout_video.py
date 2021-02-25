import argparse
import json
from pathlib import Path

import cloudpickle
import numpy as np
import torch

import rlkit.torch.pytorch_util as ptu
from multiworld.envs.pygame.point2d import Point2DEnv
from rlkit.data_management.multitask_replay_buffer import MultiTaskReplayBuffer
from rlkit.envs.images import GymEnvRenderer, EnvRenderer
# from rlkit.envs.images.text_renderer import TextRenderer
from rlkit.envs.images.plot_renderer import TextRenderer, ScrollingPlotRenderer
from rlkit.envs.pearl_envs import HalfCheetahDirEnv, AntDirEnv, PointEnv
from rlkit.envs.wrappers.flat_to_dict import FlatToDictEnv
from rlkit.misc.asset_loader import load_local_or_remote_file
from rlkit.torch.pearl.agent import MakePEARLAgentDeterministic
from rlkit.torch.pearl.buffer import PearlReplayBuffer
from rlkit.torch.pearl.diagnostics import (
    DebugInsertImagesEnv,
    FlatToDictPearlPolicy,
)
from rlkit.torch.pearl.launcher_util import load_buffer_onto_algo
from rlkit.torch.pearl.sampler import rollout, rollout_multiple_and_flatten
from rlkit.visualization.video import dump_video

counter = 0

def simulate_policy(args):
    if args.pause:
        import ipdb; ipdb.set_trace()
    # buffer_path = '/home/vitchyr/mnt2/log2/21-02-16-new-pearl--tmp-hc-create-buffer/21-02-16-new-pearl--tmp-hc-create-buffer_2021_02_16_15_08_12_id000--s22161/extra_snapshot_itr0.cpkl'
    # buffer_data = cloudpickle.load(
    #     open(buffer_path, 'rb'))
    # replay_buffer = buffer_data['replay_buffer']
    prefix = 'all_'

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
        if 'load_buffer_kwargs' in variant:
            load_buffer_kwargs = variant['load_buffer_kwargs']
            saved_tasks_path = variant['saved_tasks_path']
            task_data = load_local_or_remote_file(
            saved_tasks_path, file_type='joblib')
            tasks = task_data['tasks']
            env.wrapped_env.tasks = tasks
        else:
            load_buffer_kwargs = None
    elif 'agent' in data:  # old-style trainer
        policy = data['agent']
        env = data['env']

        variant_path = Path(snapshot_path).parent / 'variant.json'
        variant = json.load(open(variant_path, 'rb'))
        load_buffer_kwargs = variant['load_buffer_kwargs']
        saved_tasks_path = variant['saved_tasks_path']
        use_ground_truth_context = variant.get('use_ground_truth_context', False)

        task_data = load_local_or_remote_file(
            saved_tasks_path, file_type='joblib')
        tasks = task_data['tasks']
        train_task_indices = task_data['train_task_indices']
        test_task_indices = task_data['eval_task_indices']
        base_env = env.wrapped_env
        base_env.tasks = tasks
        task_indices = list(base_env.get_all_task_idx())
        if hasattr(base_env, 'task_to_vec'):
            unwrapped_tasks = [
                base_env.task_to_vec(t) for t in tasks
            ]
        else:
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
    if load_buffer_kwargs is not None:
        pearl_replay_buffer.clear_all_buffers()
        load_buffer_onto_algo(
            pearl_replay_buffer.replay_buffer,
            pearl_replay_buffer.encoder_replay_buffer,
            **load_buffer_kwargs
        )

    obs_key = 'tmp'
    policy = FlatToDictPearlPolicy(policy, obs_key)
    base_env = env.wrapped_env
    env = FlatToDictEnv(env, obs_key)

    # if isinstance(base_env, HalfCheetahDirEnv):
    #     n_tasks = 2
    #     test_task_idxs = [0, 1]
    #     n_videos = 3 * n_tasks
    #     n_repeats = 3
    # elif isinstance(base_env, AntDirEnv):
    #     n_tasks = 4
    #     n_videos = 3 * n_tasks
    #     test_task_idxs = [4, 5, 6, 7]
    #     n_repeats = 3
    #     horizon = 200
    #     img_renderer = GymEnvRenderer(width=256, height=256)
    # elif isinstance(base_env, PointEnv):
    #     n_tasks = 4
    #     n_videos = 3 * n_tasks
    #     test_task_idxs = [4, 5, 6, 7]
    #     n_repeats = 3
    #     horizon = 20
    #     img_renderer = EnvRenderer(width=256, height=256)
    # else:
    #     raise NotImplementedError()
    if isinstance(base_env, HalfCheetahDirEnv):
        n_repeats = 3
        counter_to_init_train_task = {
            0: 0,
            1: 1,
        }
        counter_to_eval_on_train_task = {
            2: 0,
            3: 1,
        }
        counter_to_eval_on_test_task = {
            4: 0,
            5: 1,
        }
        rows = 2
        columns = 3
        horizon = 200
        img_renderer = GymEnvRenderer(width=256, height=256)
    elif isinstance(base_env, AntDirEnv):
        n_repeats = 3
        counter_to_init_train_task = {
            0: 0,
            3: 1,
            6: 2,
            9: 3,
        }
        counter_to_eval_on_train_task = {
            1: 0,
            4: 1,
            7: 2,
            10: 3,
        }
        counter_to_eval_on_test_task = {
            2: 4,
            5: 5,
            8: 6,
            11: 7,
        }
        rows = 3
        columns = 4
        horizon = 200
        img_renderer = GymEnvRenderer(width=256, height=256)
    elif isinstance(base_env, PointEnv):
        n_repeats = 3
        counter_to_init_train_task = {
            0: 0,
            3: 1,
            6: 2,
            9: 3,
        }
        counter_to_eval_on_train_task = {
            1: 0,
            4: 1,
            7: 2,
            10: 3,
        }
        counter_to_eval_on_test_task = {
            2: 4,
            5: 5,
            8: 6,
            11: 7,
        }
        rows = 3
        columns = 4
        horizon = 20
        img_renderer = EnvRenderer(width=256, height=256)
    else:
        raise NotImplementedError()
    if rows * columns < (
            len(counter_to_eval_on_test_task)
            + len(counter_to_eval_on_train_task)
            + len(counter_to_init_train_task)
    ):
        print("Not all settings will be rendered. Are you sure?")
        import ipdb; ipdb.set_trace()
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
    save_path = save_dir / '{}rollout0_{}.mp4'.format(
        prefix, Path(snapshot_path).stem)

    def random_task_rollout_fn(*args, max_path_length=None, **kwargs):
        global counter
        if counter in counter_to_init_train_task:
            task_idx = counter_to_init_train_task[counter]
            init_context = pearl_replay_buffer.sample_context(task_idx)
            init_context = ptu.from_numpy(init_context)
            path = rollout_multiple_and_flatten(
                *args,
                task_idx=task_idx,
                initial_context=init_context,
                resample_latent_period=0,
                accum_context=False,
                update_posterior_period=0,
                max_path_length=int(max_path_length//n_repeats),
                num_repeats=n_repeats,
                **kwargs)
        elif counter in counter_to_eval_on_train_task:
            task_idx = counter_to_eval_on_train_task[counter]
            text_renderer.prefix = 'eval on train\n'
            path = rollout_multiple_and_flatten(
                *args,
                task_idx=task_idx,
                initial_context=None,
                resample_latent_period=0,
                accum_context=True,
                update_posterior_period=1,
                max_path_length=int(max_path_length//n_repeats),
                num_repeats=n_repeats,
                **kwargs)
        elif counter in counter_to_eval_on_test_task:
            task_idx = counter_to_eval_on_test_task[counter]
            text_renderer.prefix = 'eval on test\n'
            init_context = None
            path = rollout_multiple_and_flatten(
                *args,
                task_idx=task_idx,
                initial_context=init_context,
                resample_latent_period=0,
                accum_context=True,
                update_posterior_period=1,
                max_path_length=int(max_path_length//n_repeats),
                num_repeats=n_repeats,
                **kwargs)
        else:
            import ipdb; ipdb.set_trace()
            path = None
        counter += 1
        return path

    dump_video(
        env=img_env,
        policy=policy,
        filename=save_path,
        rollout_function=random_task_rollout_fn,
        obs_dict_key='observations',
        keys_to_show=list(renderers.keys()),
        image_format=img_renderer.output_image_format,
        imsize=256,
        rows=rows,
        columns=columns,
        horizon=horizon * n_repeats,
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
