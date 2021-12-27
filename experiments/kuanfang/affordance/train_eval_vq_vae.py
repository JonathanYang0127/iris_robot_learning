import os
# from os import path as osp
import random
from absl import app
from absl import flags
from absl import logging  # NOQA

import numpy as np
import torch
from torch.utils.data import DataLoader
# from tqdm import tqdm

import rlkit.torch.pytorch_util as ptu
from rlkit.core import logger
from lib.vae import vae_datasets
from lib.vae.vq_vae import VqVae
from lib.vae.vq_vae_trainer import VqVaeTrainer

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_string('dataset', None, 'Path to the dataset.')
flags.DEFINE_integer('seed', None, 'Random seed.')

FLAGS = flags.FLAGS


def data_loaders(train_data, test_data, batch_size):
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True)
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True)
    return train_loader, test_loader


def train_eval(
        root_dir,
        num_epochs=500,
        batch_size=64,
        dataset_path=None,
        save_interval=10,
        cached_dataset_path=False,
        trainer_kwargs=None,
        data_size=float('inf'),
        num_train_batches_per_epoch=1000,
        num_test_batches_per_epoch=10,
        dump_samples=False,
):
    logger.set_snapshot_dir(root_dir)
    logger.add_tensorboard_output(root_dir)

    trainer_kwargs = {} if trainer_kwargs is None else trainer_kwargs

    root_dir = os.path.expanduser(root_dir)
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    # dataset_ctor = vae_datasets.VaeDataset
    dataset_ctor = vae_datasets.StepVaeDataset

    train_dataset = dataset_ctor(
        dataset_path, train=True, transform=None, crop=[48, 48])
    test_dataset = dataset_ctor(
        dataset_path, train=False, transform=None, crop=[48, 48])

    train_loader, test_loader = data_loaders(
        train_dataset, test_dataset, batch_size)

    print('Finished loading data')

    # input_size = int(train_dataset[0].shape[-1])
    model = VqVae(
        # representation_size=16,
        # input_size=input_size,
        # hidden_sizes=[64, 64],
    ).to(ptu.device)

    trainer = VqVaeTrainer(
        model,
        batch_size=batch_size,
        tf_logger=logger.tensorboard_logger,
        ** trainer_kwargs,
    )

    print('Starting training')

    progress_filename = os.path.join(root_dir, 'vae_progress.csv')
    logger.add_tabular_output(progress_filename,
                              relative_to_snapshot_dir=False)

    for epoch in range(num_epochs):
        logging.info('epoch: %d' % epoch)
        should_save = (((epoch > 0) and (epoch % save_interval == 0)) or
                       epoch == num_epochs - 1)
        trainer.train_epoch(epoch, train_loader, num_train_batches_per_epoch)
        trainer.test_epoch(epoch, test_loader, num_test_batches_per_epoch)

        if should_save:
            logger.save_itr_params(epoch, model)

        stats = trainer.get_diagnostics()

        for k, v in stats.items():
            logger.record_tabular(k, v)

        logger.dump_tabular()
        trainer.end_epoch(epoch)

    logger.add_tabular_output(progress_filename,
                              relative_to_snapshot_dir=False)

    return


def main(_):
    if FLAGS.seed is not None:
        random.seed(FLAGS.seed)
        np.random.seed(FLAGS.seed)
        torch.manual_seed(FLAGS.seed)

    train_eval(FLAGS.root_dir, dataset_path=FLAGS.dataset)


if __name__ == '__main__':
    flags.mark_flag_as_required('root_dir')
    app.run(main)
