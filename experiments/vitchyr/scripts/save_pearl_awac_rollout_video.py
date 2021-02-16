import argparse
from functools import partial
from pathlib import Path

import numpy as np
import torch

import rlkit.torch.pytorch_util as ptu
from rlkit.envs.images import GymEnvRenderer
# from rlkit.envs.images.text_renderer import TextRenderer
from rlkit.envs.images.plot_renderer import TextRenderer, ScrollingPlotRenderer
from rlkit.envs.wrappers.flat_to_dict import FlatToDictPolicy, FlatToDictEnv
from rlkit.torch.pearl.diagnostics import (
    DebugInsertImagesEnv,
    FlatToDictPearlPolicy,
)
from rlkit.torch.pearl.sampler import rollout
from rlkit.visualization.video import dump_video


def simulate_policy(args):
    if args.pause:
        import ipdb; ipdb.set_trace()
    snapshot_path = args.file
    ptu.set_gpu_mode(True)
    data = torch.load(snapshot_path, map_location=ptu.device)
    policy = data['evaluation/test/posterior_live_update/policy']
    env = data['evaluation/test/posterior_live_update/env']

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
    rollout_fn = partial(
        rollout,
        accum_context=True,
        resample_latent_period=1,
        update_posterior_period=1,
        use_predicted_reward=False,
    )

    def random_task_rollout_fn(*args, **kwargs):
        task_idx = np.random.choice([0, 1])
        return rollout_fn(*args, task_idx=task_idx, **kwargs)
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
        rows=3,
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
