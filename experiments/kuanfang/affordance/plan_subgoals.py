# import os
import random
from absl import app
from absl import flags
from absl import logging  # NOQA

import matplotrlkit.experimental.kuanfang.pyplot as plt
import numpy as np
import torch

from rlkit.torch import pytorch_util as ptu

# from rlkit.experimental.kuanfang.vae import vae_datasets
from rlkit.experimental.kuanfang.planning.beam_search import BeamSearch  # NOQA
from rlkit.experimental.kuanfang.planning.random_search import RandomSearch  # NOQA
from rlkit.experimental.kuanfang.utils import io_util
from rlkit.experimental.kuanfang.utils import device_util


flags.DEFINE_string('root_dir', None, 'Root directory.')
flags.DEFINE_string('data_dir', None, 'Directory of the dataset.')
flags.DEFINE_integer('id', 0, 'Trajectory ID in the dataset.')
flags.DEFINE_integer('delta_t', 1, 'Step size to sample goals.')
flags.DEFINE_integer('max_steps', 5, 'Number of planning steps.')
flags.DEFINE_integer('seed', None, 'Random seed.')

FLAGS = flags.FLAGS


def plot_sequence(sequence):
    horizon = len(sequence)

    plt.figure(figsize=(3 * horizon, 3))

    for t in range(horizon):
        plt.subplot(1, horizon, t + 1)
        image = sequence[t] + 0.5
        image = image.permute(0, 2, 3, 1).contiguous().detach().numpy()
        plt.imshow(image[0])

    plt.show()


def compare_sequence(sequence, data, delta_t=15):
    horizon = len(sequence)

    data = ptu.from_numpy(data)

    plt.figure(figsize=(3 * horizon, 3 * 2))

    for i in range(horizon):
        plt.subplot(2, horizon, i + 1)
        t = min(i * delta_t, data.size()[0] - 1)
        image = data[t] + 0.5
        image = image.permute(1, 2, 0).contiguous()
        image = ptu.get_numpy(image)
        plt.imshow(image)

    for i in range(horizon):
        plt.subplot(2, horizon, horizon + i + 1)
        image = sequence[i] + 0.5
        image = image.permute(0, 2, 3, 1).contiguous()
        image = ptu.get_numpy(image)
        plt.imshow(image[0])

    plt.show()


def main(_):
    if FLAGS.seed is not None:
        random.seed(FLAGS.seed)
        np.random.seed(FLAGS.seed)
        torch.manual_seed(FLAGS.seed)

    device_util.set_device(True)

    logging.info('Loading the model from %s ...' % (FLAGS.root_dir))
    model = io_util.load_model(FLAGS.root_dir)

    logging.info('Loading the dataset from %s ...' % (FLAGS.data_dir))
    datasets = io_util.load_datasets(FLAGS.data_dir, keys=['test'])

    logging.info('Sampling the sequence...')
    data = datasets['test'].data[FLAGS.id]

    # search_model = BeamSearch(model)
    # search_model.set_and_encode_prior_data(data)

    search_model = RandomSearch(model)

    s_0 = data[0]
    s_g = data[min(FLAGS.delta_t * FLAGS.max_steps, data.shape[0] - 1)]
    sequence = search_model(s_0, s_g, FLAGS.max_steps)

    logging.info('Plotting the sequence...')
    # plot_sequence(sequence)
    compare_sequence(sequence, data, delta_t=FLAGS.delta_t)


if __name__ == '__main__':
    flags.mark_flag_as_required('root_dir')
    app.run(main)
