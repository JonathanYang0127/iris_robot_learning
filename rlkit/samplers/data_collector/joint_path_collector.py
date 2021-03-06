from collections import OrderedDict
from typing import Dict

from rlkit.core.logging import add_prefix
from rlkit.samplers.data_collector import PathCollector
import math


class JointPathCollector(PathCollector):
    EVENLY = 'evenly'

    def __init__(
            self,
            path_collectors: Dict[str, PathCollector],
            divide_num_steps_strategy=EVENLY,
    ):
        """
        :param path_collectors: Dictionary of path collectors
        :param divide_num_steps_strategy: How the steps are divided among the
        path collectors.
        Valid values:
         - 'evenly': divide `num_steps' evenly among the collectors
         - Dict[str, float]: divide `num_steps` according to the fractions, which must sum to one.
        """
        sorted_collectors = OrderedDict()
        # Sort the path collectors to have a canonical ordering
        for k in sorted(path_collectors):
            sorted_collectors[k] = path_collectors[k]
        self.path_collectors = sorted_collectors
        self.divide_num_steps_strategy = divide_num_steps_strategy
        if isinstance(divide_num_steps_strategy, dict):
            if abs(sum(divide_num_steps_strategy.values()) - 1.0) > 1e-10:
                raise ValueError("values must sum to 1")
            if set(divide_num_steps_strategy.keys()) != set(path_collectors.keys()):
                raise ValueError("Must give fraction for each path collector.")
        elif divide_num_steps_strategy not in {self.EVENLY}:
            raise ValueError(divide_num_steps_strategy)

    def collect_new_paths(self, max_path_length, num_steps,
                          discard_incomplete_paths,
                          **kwargs):
        paths = []
        for name, collector in self.path_collectors.items():
            paths += collector.collect_new_paths(
                max_path_length=max_path_length,
                num_steps=self._get_num_steps(num_steps, name),
                discard_incomplete_paths=discard_incomplete_paths,
                **kwargs
            )
        return paths

    def _get_num_steps(self, total_num_steps, name):
        if self.divide_num_steps_strategy == self.EVENLY:
            return total_num_steps // len(self.path_collectors)
        elif isinstance(self.divide_num_steps_strategy, dict):
            return math.ceil(total_num_steps * self.divide_num_steps_strategy[name])
        else:
            raise ValueError(self.divide_num_steps_strategy)

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
