import random
from absl import app
from absl import flags
from absl import logging  # NOQA

import matplotlib.pyplot as plt
import numpy as np
import torch

from rlkit.torch import pytorch_util as ptu

from lib.utils import io_util

from lib.utils import device_util


flags.DEFINE_string('root_dir', None, 'Root directory.')
flags.DEFINE_string('data_dir', None, 'Directory of the dataset.')
flags.DEFINE_integer('id', 0, 'Trajectory ID in the dataset.')
flags.DEFINE_integer('delta_t', 1, 'Step size to sample goals.')
flags.DEFINE_integer('max_steps', 5, 'Number of planning steps.')
flags.DEFINE_integer('seed', None, 'Random seed.')

FLAGS = flags.FLAGS


def disturb_and_plot_transitions(model, data, num_samples=6):
    noise_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0]

    vqvae = model['vqvae']
    affordance = model['affordance']
    # classifier = model['classifier']

    data = ptu.from_numpy(data)

    s = data.view(
        -1,
        vqvae.input_channels,
        vqvae.imsize,
        vqvae.imsize).contiguous()

    s = s[:2]
    s0 = s[0:1]
    s1 = s[1:2]
    h = vqvae.encode(s, flatten=False)
    h0 = h[0:1]
    h1 = h[1:2]

    z_mu, z_logvar = affordance.encode(h1, cond=h0)

    std = z_logvar.mul(0.5).exp_()
    eps = std.data.new(std.size()).normal_()
    z = eps.mul(std).add_(z_mu)

    plt.figure(figsize=(3 * len(noise_levels), 3 * (2 + num_samples)))

    for j, nl in enumerate(noise_levels):
        # offset = j * (2 + num_samples)
        stride = len(noise_levels)

        plt.subplot(2 + num_samples, len(noise_levels), j + 0 * stride + 1)
        image = s0[0] + 0.5
        image = image.permute(1, 2, 0).contiguous()
        image = ptu.get_numpy(image)
        plt.imshow(image)
        # plt.axis('off')

        plt.subplot(2 + num_samples, len(noise_levels), j + 1 * stride + 1)
        image = s1[0] + 0.5
        image = image.permute(1, 2, 0).contiguous()
        image = ptu.get_numpy(image)
        plt.imshow(image)
        # plt.axis('off')

        for i in range(num_samples):
            noisy_z = z + nl * ptu.randn(1, affordance.representation_size)
            h1_pred = affordance.decode(noisy_z, cond=h0).detach()
            s1_pred = vqvae.decode(h1_pred).detach()

            plt.subplot(2 + num_samples, len(noise_levels),
                        j + (2 + i) * stride + 1)
            image = s1_pred + 0.5
            image = image[0]
            image = image.permute(1, 2, 0).contiguous()
            image = ptu.get_numpy(image)
            plt.imshow(image)
            # plt.axis('off')

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
    data = datasets['test'].data[FLAGS.id][::FLAGS.delta_t]
    disturb_and_plot_transitions(model, data)


if __name__ == '__main__':
    flags.mark_flag_as_required('root_dir')
    app.run(main)
