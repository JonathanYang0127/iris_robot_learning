import random
from absl import app
from absl import flags
from absl import logging  # NOQA

import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.special import comb

from rlkit.torch import pytorch_util as ptu

from lib.vae import vae_datasets
from lib.utils import io_util
from lib.utils import device_util


flags.DEFINE_string('root_dir', None, 'Root directory.')
flags.DEFINE_string('data_dir', None, 'Path to the data.')
# flags.DEFINE_integer('id', 0, 'Trajectory ID in the dataset.')
flags.DEFINE_integer('num_samples', 4096, 'Number of samples.')
flags.DEFINE_integer('delta_t', 15, 'Step size to sample goals.')
flags.DEFINE_integer('seed', None, 'Random seed.')

FLAGS = flags.FLAGS


def encode_and_visualize(
        model,
        data,
        boundary=5,
):
    vqvae = model['vqvae']
    affordance = model['affordance']
    classifier = model['classifier']  # NOQA
    discriminator = model['discriminator']  # NOQA

    s_0 = data[:, 0]
    s_0 = ptu.from_numpy(s_0)
    h_0 = vqvae.encode(s_0, flatten=False)

    s_1 = data[:, 1]
    s_1 = ptu.from_numpy(s_1)
    h_1 = vqvae.encode(s_1, flatten=False)

    print(h_0.get_device())
    print(h_1.get_device())
    z, _ = affordance.encode(h_1, cond=h_0)
    z = ptu.get_numpy(z)

    ########################################
    # Plot.
    ########################################
    dim_z = int(z.shape[-1])
    num_plots = comb(dim_z, 2)
    num_plots_root = int(np.ceil(np.sqrt(num_plots)))
    plt.figure(figsize=(5 * num_plots_root, 5 * num_plots_root))
    plt.title('Visualization of Affordance Encodings')
    plot_id = 0
    for i in range(dim_z):
        for j in range(i + 1, dim_z):

            plt.subplot(num_plots_root, num_plots_root, plot_id + 1)
            plt.title('(%d, %d)' % (i, j))
            plt.xlim(-boundary, boundary)
            plt.ylim(-boundary, boundary)
            plt.scatter(
                z[:, i],
                z[:, j],
                # s=5,
                # c='b',
                # alpha=0.5,
            )

            plot_id += 1


def main(_):
    if FLAGS.seed is not None:
        random.seed(FLAGS.seed)
        np.random.seed(FLAGS.seed)
        torch.manual_seed(FLAGS.seed)

    device_util.set_device(True)

    logging.info('Loading the model from %s ...' % (FLAGS.root_dir))
    model = io_util.load_model(FLAGS.root_dir)

    logging.info('Loading the dataset from %s ...' % (FLAGS.data_dir))
    datasets = io_util.load_datasets(
        data_dir=FLAGS.data_dir,
        dataset_ctor=vae_datasets.VaeGoalDataset,
        keys=['test'],
    )
    test_dataset = datasets['test']

    data = test_dataset.data[:FLAGS.num_samples, ::FLAGS.delta_t]

    # id = 0
    # data = np.concatenate(
    #     [test_dataset.data[id, :-FLAGS.delta_t],
    #      test_dataset.data[id, FLAGS.delta_t:]],
    #     axis=1)

    logging.info('Encoding and visualizing the transitions...')
    encode_and_visualize(model, data)

    plt.show()


if __name__ == '__main__':
    flags.mark_flag_as_required('root_dir')
    app.run(main)
