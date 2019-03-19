from collections import OrderedDict

import gtimer as gt

from railrl.core import logger
from railrl.data_management.replay_buffer import ReplayBuffer
from railrl.misc import eval_util
from railrl.samplers.data_collector import PathCollector


def _get_epoch_timings(epoch):
    times_itrs = gt.get_times().stamps.itrs
    train_time = times_itrs['training'][-1]
    expl_sampling_time = times_itrs['exploration sampling'][-1]
    data_storing_time = times_itrs['data storing'][-1]
    save_time = times_itrs['saving'][-1]
    eval_sampling_time = times_itrs['evaluation sampling'][-1] if epoch > 0 else 0
    epoch_time = train_time + expl_sampling_time + eval_sampling_time
    total_time = gt.get_times().total

    return OrderedDict([
        ('time/data storing (s)', data_storing_time),
        ('time/training (s)', train_time),
        ('time/evaluation sampling (s)', eval_sampling_time),
        ('time/exploration sampling (s)', expl_sampling_time),
        ('time/saving (s)', save_time),
        ('time/epoch (s)', epoch_time),
        ('time/total train (s)', total_time),
    ])


class BatchRLAlgorithm(object):
    def __init__(
            self,
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector: PathCollector,
            evaluation_data_collector: PathCollector,
            data_buffer: ReplayBuffer,
            batch_size,
            max_path_length,
            num_epochs,
            num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
            num_train_loops_per_epoch=1,
            min_num_steps_before_training=0,
    ):
        self.trainer = trainer
        self.expl_env = exploration_env
        self.eval_env = evaluation_env
        self.expl_data_collector = exploration_data_collector
        self.eval_data_collector = evaluation_data_collector
        self.data_buffer = data_buffer
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training
        self._start_epoch = 0

    def train(self, start_epoch=0):
        self._start_epoch = start_epoch
        self._train()

    def _train(self):
        if self.min_num_steps_before_training > 0:
            init_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.min_num_steps_before_training,
            )
            self.data_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)

        for epoch in gt.timed_for(
                range(self._start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            self.eval_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_eval_steps_per_epoch,
            )
            gt.stamp('evaluation sampling')

            for _ in range(self.num_train_loops_per_epoch):
                new_expl_paths = self.expl_data_collector.collect_new_paths(
                    self.max_path_length,
                    self.num_expl_steps_per_train_loop,
                )
                gt.stamp('exploration sampling', unique=False)

                self.data_buffer.add_paths(new_expl_paths)
                gt.stamp('data storing', unique=False)

                for _ in range(self.num_trains_per_train_loop):
                    train_data = self.data_buffer.random_batch(self.batch_size)
                    self.trainer.train(train_data)
                gt.stamp('training', unique=False)

            self._end_epoch(epoch)

    def _end_epoch(self, epoch):
        snapshot = {}
        for k, v in self.trainer.get_snapshot().items():
            snapshot['trainer/' + k] = v
        for k, v in self.expl_data_collector.get_snapshot().items():
            snapshot['exploration/' + k] = v
        for k, v in self.eval_data_collector.get_snapshot().items():
            snapshot['evaluation/' + k] = v
        for k, v in self.data_buffer.get_snapshot().items():
            snapshot['buffer/' + k] = v
        logger.save_itr_params(epoch, snapshot)
        gt.stamp('saving')

        self._log_stats(epoch)

        self.expl_data_collector.end_epoch(epoch)
        self.eval_data_collector.end_epoch(epoch)
        self.data_buffer.end_epoch(epoch)
        self.trainer.end_epoch(epoch)

    def _log_stats(self, epoch):
        logger.log("Epoch {} finished".format(epoch), with_timestamp=True)

        """
        Data Buffer
        """
        logger.record_dict(self.data_buffer.get_diagnostics(), prefix='buffer/')

        """
        Trainer
        """
        logger.record_dict(self.trainer.get_diagnostics(), prefix='trainer/')

        """
        Exploration
        """
        logger.record_dict(
            self.expl_data_collector.get_diagnostics(),
            prefix='exploration/'
        )
        expl_paths = self.expl_data_collector.get_epoch_paths()
        if hasattr(self.expl_env, 'get_diagnostics'):
            logger.record_dict(
                self.expl_env.get_diagnostics(expl_paths),
                prefix='exploration/',
            )
        logger.record_dict(
            eval_util.get_generic_path_information(expl_paths),
            prefix="exploration/",
        )
        """
        Evaluation
        """
        logger.record_dict(
            self.eval_data_collector.get_diagnostics(),
            prefix='evaluation/',
        )
        eval_paths = self.eval_data_collector.get_epoch_paths()
        if hasattr(self.eval_env, 'get_diagnostics'):
            logger.record_dict(
                self.eval_env.get_diagnostics(eval_paths),
                prefix='evaluation/',
            )
        logger.record_dict(
            eval_util.get_generic_path_information(eval_paths),
            prefix="evaluation/",
        )

        """
        Misc
        """
        logger.record_dict(_get_epoch_timings(epoch))
        logger.record_tabular('Epoch', epoch)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)
