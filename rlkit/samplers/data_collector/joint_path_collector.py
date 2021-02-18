from collections import OrderedDict
from typing import Dict

from rlkit.core.logging import add_prefix
from rlkit.samplers.data_collector import PathCollector


class JointPathCollector(PathCollector):
    def __init__(self, path_collectors: Dict[str, PathCollector]):
        sorted_collectors = OrderedDict()
        # Sort the path collectors to have a canonical ordering
        for k in sorted(path_collectors):
            sorted_collectors[k] = path_collectors[k]
        self.path_collectors = sorted_collectors

    def collect_new_paths(self, max_path_length, num_steps,
                          discard_incomplete_paths,
                          **kwargs):
        paths = []
        for name, collector in self.path_collectors.items():
            paths += collector.collect_new_paths(
                max_path_length=max_path_length,
                num_steps=num_steps,
                discard_incomplete_paths=discard_incomplete_paths,
                **kwargs
            )
        return paths

    def end_epoch(self, epoch):
        for collector in self.path_collectors.values():
            collector.end_epoch(epoch)

    def get_diagnostics(self):
        diagnostics = OrderedDict()
        num_steps = 0
        num_paths = 0
        for name, collector in self.path_collectors.items():
            stats = collector.get_diagnostics()
            num_steps += stats['num steps total']
            num_paths += stats['num paths total']
            diagnostics.update(
                add_prefix(stats, name, divider='/'),
            )
        diagnostics['num steps total'] = num_steps
        diagnostics['num paths total'] = num_paths
        return diagnostics

    def get_snapshot(self):
        snapshot = {}
        for name, collector in self.path_collectors.items():
            snapshot.update(
                add_prefix(collector.get_snapshot(), name, divider='/'),
            )
        return snapshot

    def get_epoch_paths(self):
        paths = {}
        for name, collector in self.path_collectors.items():
            paths[name] = collector.get_epoch_paths()
        return paths
