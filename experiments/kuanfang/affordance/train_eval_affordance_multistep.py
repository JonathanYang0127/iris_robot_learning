import os
import random
from absl import app
from absl import flags
from absl import logging  # NOQA

import gin
import numpy as np
import torch

import rlkit.torch.pytorch_util as ptu
from rlkit.core import logger
from rlkit.experimental.kuanfang.vae import vae_datasets

from rlkit.experimental.kuanfang.vae import affordance_networks

from rlkit.experimental.kuanfang.vae.vqvae import VqVae
from rlkit.experimental.kuanfang.vae.affordance_trainer_multistep import AffordanceTrainer  # NOQA
from rlkit.experimental.kuanfang.utils import io_util
from rlkit.experimental.kuanfang.utils import device_util

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_string('data_dir', None, 'Path to the data.')
flags.DEFINE_string('vqvae', None, 'Path to the pretrained vqvae.')
flags.DEFINE_integer('delta_t', 1, 'Step size to sample goals.')
flags.DEFINE_multi_string(
    'gin_file', None, 'Path to the trainer config files.')
flags.DEFINE_multi_string('gin_param', None, 'Gin binding to pass through.')
flags.DEFINE_integer('seed', None, 'Random seed.')

FLAGS = flags.FLAGS


@gin.configurable  # NOQA
def train_eval(
        root_dir,
        num_epochs=10000,
        batch_size=64,
        data_dir=None,
        pretrained_vqvae_dir=None,
        save_interval=100,
        trainer_kwargs=None,
        delta_t=None,
        num_train_batches_per_epoch=100,
        num_test_batches_per_epoch=1,
        dump_samples=False,
        # Model parameters.
        embedding_dim=5,

        z_dim=4,

        affordance_pred_weight=1000.,
        affordance_beta=1.0,
        # wgan_gen_weight=0.,
        # wgan_clip_value=0.01,

        # TODO
        # image_dist_thresh=None,
        image_dist_thresh=15.,
):
    device_util.set_device(True)

    logger.set_snapshot_dir(root_dir)
    logger.add_tensorboard_output(root_dir)

    vqvae_path = os.path.join(root_dir, 'vqvae.pt')
    affordance_path = os.path.join(root_dir, 'affordance.pt')
    classifier_path = os.path.join(root_dir, 'classifier.pt')

    trainer_kwargs = {} if trainer_kwargs is None else trainer_kwargs

    root_dir = os.path.expanduser(root_dir)
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    if pretrained_vqvae_dir is not None:
        pretrained_vqvae_path = os.path.join(pretrained_vqvae_dir, 'vqvae.pt')
        print('Loading the pretrained VQVAE from %s...'
              % (pretrained_vqvae_path))
        vqvae = torch.load(pretrained_vqvae_path)
        torch.save(vqvae, vqvae_path)
        use_pretrained_vqvae = True
    else:
        vqvae = VqVae(
            embedding_dim=embedding_dim,
        ).to(ptu.device)
        use_pretrained_vqvae = False

    affordance = affordance_networks.CcVae(
        data_channels=embedding_dim,
        z_dim=z_dim,
    ).to(ptu.device)

    classifier = affordance_networks.Classifier(
        data_channels=embedding_dim,
    ).to(ptu.device)
    # classifier = None  # TODO

    datasets = io_util.load_datasets(
        data_dir=data_dir,
        encoding_dir=pretrained_vqvae_dir,
        dataset_ctor=vae_datasets.VaeMultistepDataset,
        vqvae_mode='zq',
        transform=None,
        crop=[48, 48],
        delta_t=delta_t,
    )
    train_dataset = datasets['train']
    test_dataset = datasets['test']

    if pretrained_vqvae_dir is not None:
        assert train_dataset.encoding is not None
        assert test_dataset.encoding is not None

    train_loader, test_loader = io_util.data_loaders(
        train_dataset, test_dataset, batch_size)

    print('Finished loading data')

    trainer = AffordanceTrainer(
        vqvae=vqvae,
        affordance=affordance,
        classifier=classifier,
        use_pretrained_vqvae=use_pretrained_vqvae,
        tf_logger=logger.tensorboard_logger,
        affordance_pred_weight=affordance_pred_weight,
        affordance_beta=affordance_beta,
        # wgan_gen_weight=wgan_gen_weight,
        # wgan_clip_value=wgan_clip_value,
        image_dist_thresh=image_dist_thresh,
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
            logging.info('Saving the model to %s...' % (root_dir))

            if pretrained_vqvae_dir is None:
                torch.save(vqvae, vqvae_path)

            torch.save(affordance, affordance_path)

            if classifier is not None:
                torch.save(classifier, classifier_path)

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

    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)

    train_eval(FLAGS.root_dir,
               data_dir=FLAGS.data_dir,
               pretrained_vqvae_dir=FLAGS.vqvae,
               delta_t=FLAGS.delta_t,
               )


if __name__ == '__main__':
    flags.mark_flag_as_required('root_dir')
    app.run(main)
