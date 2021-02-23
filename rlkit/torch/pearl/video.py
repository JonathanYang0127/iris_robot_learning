import os.path as osp

from rlkit.core import logger
from rlkit.torch.pearl.path_collector import (
    PearlJointPathCollector,
)
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

    def __call__(self, algo, epoch):
        logdir = logger.get_snapshot_dir()
        if epoch % self.save_video_period == 0 or epoch >= algo.num_epochs - 1:
            def label_name(name, kwargs):
                self.text_renderer.prefix = '{name}\n'.format(name=name)
                if self.task_indices_per_rollout is not None:
                    kwargs['task_indices_for_rollout'] = self.task_indices_per_rollout
                return kwargs
            name_to_path_and_indices = self.path_collector.collect_named_paths_and_indices(
                self.max_path_length,
                self.num_steps,
                False,
                per_name_callback=label_name,
            )
            for name, (paths, _) in name_to_path_and_indices.items():
                filename = 'video_{tag}{name}_{epoch}.mp4'.format(
                    tag=self.tag,
                    name=name.replace('/', '-'),
                    epoch=epoch)
                filepath = osp.join(logdir, filename)
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
