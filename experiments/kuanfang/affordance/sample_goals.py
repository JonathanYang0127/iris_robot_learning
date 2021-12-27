# import os
import random
from absl import app
from absl import flags
from absl import logging  # NOQA

import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sn
# import pandas as pd
import torch

from rlkit.torch import pytorch_util as ptu

# from lib.vae import vae_datasets
from lib.utils import io_util
from lib.utils import device_util


flags.DEFINE_string('root_dir', None, 'Root directory.')
flags.DEFINE_string('data_dir', None, 'Path to the data.')
flags.DEFINE_integer('id', 0, 'Trajectory ID in the dataset.')
flags.DEFINE_integer('t', 0, 'Initial time step.')
flags.DEFINE_integer('delta_t', 15, 'Step size to sample goals.')
flags.DEFINE_integer('seed', None, 'Random seed.')

FLAGS = flags.FLAGS


def plot_sequence(data):
    horizon = data.shape[0]

    plt.figure(figsize=(3 * horizon, 3))

    for t in range(horizon):
        plt.subplot(1, horizon, t + 1)
        image = data[t] + 0.5
        image = np.transpose(image, (1, 2, 0))
        plt.title('t%02d' % (t))
        plt.imshow(image)


def sample_topk(model,
                data,
                # num_samples=1024,
                num_samples=2048,
                topk=25,
                mode='random',
                # mode='top',
                # mode='nms',
                nms_thresh=5.0,
                ):
    vqvae = model['vqvae']
    affordance = model['affordance']
    classifier = model['classifier']  # NOQA
    discriminator = model['discriminator']  # NOQA

    s_0 = data[0]
    s_0 = ptu.from_numpy(s_0[np.newaxis, :, :, :])
    h_0 = vqvae.encode(s_0, flatten=False)
    h_0 = torch.stack([h_0[0]] * num_samples, dim=0)

    # Sample latent codes.
    z = affordance.sample_prior(num_samples)
    z = ptu.from_numpy(z)
    h_samp = affordance.decode(z, cond=h_0)

    # Classifiy.
    logits = classifier(h_0, h_samp)
    # logits = discriminator(h_0, h_samp)

    logits = torch.squeeze(logits, -1)
    print(logits)
    print(logits.shape)

    # Select.
    if mode == 'random':
        top_h_samp = h_samp
        top_logits = logits

    elif mode == 'top':
        top_logits, top_indices = torch.topk(logits, topk)
        print(top_indices.shape, top_indices.dtype)

        top_h_samp = h_samp[top_indices]

    elif mode == 'nms':
        top_logits, top_indices = torch.topk(logits, num_samples)
        print(top_indices.shape, top_indices.dtype)

        diffs = torch.unsqueeze(h_samp, 0) - torch.unsqueeze(h_samp, 1)
        diffs = diffs.view(num_samples, num_samples, -1)
        dists = torch.norm(diffs, dim=-1)
        # dists = ptu.get_numpy(dists)
        print('dists: ', dists)

        valids = torch.ones((num_samples, ), dtype=torch.float32
                            ).to(ptu.device)
        selected_inds = []
        for i_top in range(num_samples):
            if len(selected_inds) >= topk:
                break

            ind = top_indices[i_top]

            # print('-- Select: ', i_top, ind, valids[ind])
            # print(dists[ind] >= nms_thresh)

            if valids[ind] == 0:
                continue

            selected_inds.append(ind)

            print((dists[ind] >= nms_thresh).dtype)
            valids = torch.where((dists[ind] >= nms_thresh).to(torch.bool),
                                 valids,
                                 torch.zeros_like(valids))

        selected_inds = np.array(selected_inds, dtype=np.int64)
        selected_inds = ptu.from_numpy(selected_inds).to(torch.int64)

        print(selected_inds.shape, selected_inds.dtype, h_samp.shape)
        top_h_samp = h_samp[selected_inds]
        top_z = z[selected_inds]

        print('-- top z --')
        print(top_z)

    # VQVAE Decode.
    top_s_samp = vqvae.decode(top_h_samp)
    top_s_samp = ptu.get_numpy(top_s_samp)

    ########################################
    # Plot.
    ########################################
    topk_root = int(np.ceil(np.sqrt(topk)))
    plt.figure(figsize=(5 * topk_root, 5 * topk_root))
    plt.title('Sampled Goals')
    for i in range(topk_root):
        for j in range(topk_root):
            ind = i * topk_root + j

            if ind >= top_s_samp.shape[0]:
                logging.warn('There are only %d samples.', top_s_samp.shape[0])
                return

            plt.subplot(topk_root, topk_root, ind + 1)
            image = top_s_samp[ind] + 0.5
            image = np.transpose(image, (1, 2, 0))
            plt.imshow(image)
            score = top_logits[ind]
            plt.title('%03f' % (score))


def main(_):
    if FLAGS.seed is not None:
        random.seed(FLAGS.seed)
        np.random.seed(FLAGS.seed)
        torch.manual_seed(FLAGS.seed)

    device_util.set_device(False)

    logging.info('Loading the model from %s ...' % (FLAGS.root_dir))
    model = io_util.load_model(FLAGS.root_dir)

    logging.info('Loading the dataset from %s ...' % (FLAGS.data_dir))
    datasets = io_util.load_datasets(FLAGS.data_dir, keys=['test'])
    test_dataset = datasets['test']
    data = test_dataset.data[FLAGS.id, FLAGS.t::FLAGS.delta_t]

    logging.info('Plotting the sequence...')
    plot_sequence(data)

    logging.info('Sampling goals...')
    sample_topk(model, data)

    plt.show()


if __name__ == '__main__':
    flags.mark_flag_as_required('root_dir')
    app.run(main)
