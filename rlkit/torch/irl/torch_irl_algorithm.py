from collections import OrderedDict

from rlkit.core.timer import timer

from rlkit.core import logger
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.misc import eval_util
from rlkit.samplers.data_collector.path_collector import PathCollector
from rlkit.core.batch_rl_algorithm import BatchRLAlgorithm

from rlkit.core.logging import append_log
import rlkit.torch.pytorch_util as ptu

class TorchIRLAlgorithm(BatchRLAlgorithm):
    def __init__(
            self,
            reward_trainer,
            *args,
            **kwargs
    ):
        super().__init__(
            *args,
            **kwargs
        )
        self.reward_trainer = reward_trainer

    def _train(self):
        done = (self.epoch == self.num_epochs)
        if done:
            return OrderedDict(), done

        if self.epoch == 0 and self.min_num_steps_before_training > 0:
            init_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
            )
            self.replay_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)

        timer.start_timer('evaluation sampling')
        if self.epoch % self._eval_epoch_freq == 0:
            self.eval_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_eval_steps_per_epoch,
                discard_incomplete_paths=True,
            )
        timer.stop_timer('evaluation sampling')

        if not self._eval_only:
            for _ in range(self.num_train_loops_per_epoch):
                timer.start_timer('exploration sampling', unique=False)
                new_expl_paths = self.expl_data_collector.collect_new_paths(
                    self.max_path_length,
                    self.num_expl_steps_per_train_loop,
                    discard_incomplete_paths=False,
                )
                timer.stop_timer('exploration sampling')

                timer.start_timer('replay buffer data storing', unique=False)
                self.replay_buffer.add_paths(new_expl_paths)
                timer.stop_timer('replay buffer data storing')

                timer.start_timer('training', unique=False)
                # self.reward_trainer.model.cuda()
                # self.trainer.policy.cuda()
                # ptu.set_gpu_mode(True)
                for _ in range(self.num_trains_per_train_loop):
                    train_data = self.replay_buffer.random_batch(self.batch_size)
                    self.trainer.train(train_data)
                    self.reward_trainer.train(train_data)
                # self.reward_trainer.model.cpu()
                # self.trainer.policy.cpu()
                # ptu.set_gpu_mode(False)
                timer.stop_timer('training')
        log_stats = self._get_diagnostics()
        return log_stats, False

    def _get_diagnostics(self):
        algo_log = super()._get_diagnostics()
        append_log(algo_log, self.reward_trainer.get_diagnostics(),
                   prefix='reward_trainer/')
        return algo_log

    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)

    def training_mode(self, mode):
        for net in self.trainer.networks:
            net.train(mode)