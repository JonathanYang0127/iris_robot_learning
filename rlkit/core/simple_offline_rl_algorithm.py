import time

from rlkit.core import logger
from rlkit.core.logging import add_prefix
from rlkit.torch.core import np_to_pytorch_batch


class SimpleOfflineRlAlgorithm(object):
    def __init__(self, trainer, replay_buffer, batch_size, logging_period, num_batches):
        self.trainer = trainer
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.logging_period = logging_period

    def pretrain_q_with_bc_data(self):
        logger.remove_tabular_output(
            'progress.csv', relative_to_snapshot_dir=True
        )
        logger.add_tabular_output(
            'pretrain.csv', relative_to_snapshot_dir=True
        )
        # first train only the Q function
        for i in range(self.num_batches):
            train_data = self.replay_buffer.random_batch(self.batch_size)
            train_data = np_to_pytorch_batch(train_data)
            obs = train_data['observations']
            next_obs = train_data['next_observations']
            train_data['observations'] = obs
            train_data['next_observations'] = next_obs
            self.trainer.train_from_torch(train_data, pretrain=True)
            if i % self.logging_period == 0:
                stats_with_prefix = add_prefix(
                    self.trainer.eval_statistics, prefix="trainer/")
                logger.record_dict(stats_with_prefix)
                logger.dump_tabular(with_prefix=True, with_timestamp=False)
