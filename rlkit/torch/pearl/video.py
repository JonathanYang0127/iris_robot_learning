from collections import OrderedDict
import os.path as osp
from pathlib import Path
from typing import Callable, List

from gym import Env

from rlkit.core import logger
from rlkit.envs.images import GymEnvRenderer, EnvRenderer
from rlkit.envs.images.plot_renderer import TextRenderer, ScrollingPlotRenderer
from rlkit.envs.pearl_envs import PointEnv
from rlkit.envs.wrappers.flat_to_dict import FlatToDictEnv
from rlkit.policies.base import Policy
from rlkit.torch.pearl.diagnostics import (
    DebugInsertImagesEnv,
    FlatToDictPearlPolicy,
)
from rlkit.torch.pearl.path_collector import (
    PearlJointPathCollector,
)
from rlkit.torch.pearl.sampler import merge_paths
from rlkit.visualization.video import dump_paths


class PearlSaveVideoFunction(object):
    def __init__(
            self,
            path_collector: PearlJointPathCollector,
            save_video_period,
            keys_to_save,
            max_path_length,
            num_steps,
            text_renderer,
            tag='',
            task_indices_per_rollout=None,
            logdir=None,
            discard_incomplete_paths=False,
            flatten_paths=False,
            num_rollout_per_setting=3,
            **dump_video_kwargs
    ):
        self.path_collector = path_collector
        self.save_video_period = save_video_period
        self.keys_to_save = keys_to_save
        self.max_path_length = max_path_length
        self.num_steps = num_steps
        self.text_renderer = text_renderer
        self.tag = tag + '_' if tag else ''
        self.task_indices_per_rollout = task_indices_per_rollout
        self.dump_video_kwargs = dump_video_kwargs
        self.logdir = logdir or logger.get_snapshot_dir()
        self.discard_incomplete_paths = discard_incomplete_paths
        self.flatten_paths = flatten_paths
        self.num_rollout_per_setting = num_rollout_per_setting
        Path(self.logdir).mkdir(parents=True, exist_ok=True)

    def __call__(self, algo, epoch):
        if epoch % self.save_video_period == 0 or epoch >= algo.num_epochs - 1:
            def label_name(name, kwargs):
                self.text_renderer.prefix = '{name}\n'.format(name=name)
                if self.task_indices_per_rollout is not None:
                    kwargs['task_indices_for_rollout'] = self.task_indices_per_rollout
                return kwargs
            name_to_path_and_indices = self.path_collector.yield_name_paths_and_indices(
                self.max_path_length,
                self.num_steps,
                self.discard_incomplete_paths,
                per_name_callback=label_name,
                flatten_paths=self.flatten_paths,
            )
            for name, paths, _ in name_to_path_and_indices:
                filename = 'video_{tag}{name}_{epoch}.mp4'.format(
                    tag=self.tag,
                    name=name.replace('/', '-'),
                    epoch=epoch)
                filepath = osp.join(self.logdir, filename)
                dump_paths(
                    None,
                    filepath,
                    paths,
                    keys=self.keys_to_save,
                    columns=len(self.task_indices_per_rollout),
                    rows=None,  # fill in automatically
                    # columns=len(video_path_collector.path_collectors),
                    # columns=self
                    **self.dump_video_kwargs
                )


def make_save_video_function(
        env: Env,
        policy: Policy,
        tag: str,
        create_path_collector: Callable[[Env, Policy], PearlJointPathCollector],
        num_steps: int,
        task_indices: List[int],
        max_path_length: int,
        video_img_size: int,
        **save_video_kwargs
):
    font_size = int(video_img_size / 256 * 40)  # heuristic

    def config_return_ax(ax):
        ax.set_title('returns vs step')
        ax.set_xlabel('steps')
        ax.set_ylabel('returns')
        size = font_size
        ax.yaxis.set_tick_params(labelsize=size)
        ax.xaxis.set_tick_params(labelsize=size)
        ax.title.set_size(size)
        ax.xaxis.label.set_size(size)
        ax.yaxis.label.set_size(size)

    obs_key = 'obervation_for_video'
    img_policy = FlatToDictPearlPolicy(policy, obs_key)
    base_env = env.wrapped_env
    env = FlatToDictEnv(env, obs_key)

    if isinstance(base_env, PointEnv):
        img_renderer = EnvRenderer(
            width=video_img_size,
            height=video_img_size,
        )
    else:
        img_renderer = GymEnvRenderer(
            width=video_img_size,
            height=video_img_size,
        )
    text_renderer = TextRenderer(
        text='test',
        width=video_img_size,
        height=video_img_size,
        font_size=font_size,
    )
    returns_plotter = ScrollingPlotRenderer(
        width=video_img_size,
        height=video_img_size,
        modify_ax_fn=config_return_ax,
        window_size=min(max_path_length, 200),
        cumsum=True,
    )
    renderers = OrderedDict([
        ('image_observation', img_renderer),
        ('return_plot', returns_plotter),
        ('text', text_renderer),
    ])
    img_env = DebugInsertImagesEnv(
        wrapped_env=env,
        renderers=renderers,
    )
    video_path_collector = create_path_collector(img_env, img_policy)
    keys_to_save = list(renderers.keys())
    return PearlSaveVideoFunction(
        video_path_collector,
        keys_to_save=keys_to_save,
        obs_dict_key='observations',
        image_format=text_renderer.output_image_format,
        text_renderer=text_renderer,
        imsize=video_img_size,
        unnormalize=True,
        task_indices_per_rollout=task_indices,
        tag=tag,
        num_steps=num_steps,
        max_path_length=max_path_length,
        **save_video_kwargs
    )