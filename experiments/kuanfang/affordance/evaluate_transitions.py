import random
from absl import app
from absl import flags
from absl import logging  # NOQA

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import torch

from rlkit.torch import pytorch_util as ptu

from lib.utils import io_util
from lib.utils import device_util


flags.DEFINE_string('root_dir', None, 'Root directory.')
flags.DEFINE_string('data_dir', None, 'Path to the data.')
flags.DEFINE_integer('id', 0, 'Trajectory ID in the dataset.')
flags.DEFINE_integer('delta_t', 1, 'Step size to sample goals.')
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


def evaluate_transitions(model, data):
    vqvae = model['vqvae']
    affordance = model['affordance']
    classifier = model['classifier']

    horizon = data.shape[0]
    data = ptu.from_numpy(data)

    s = data.view(
        -1,
        vqvae.input_channels,
        vqvae.imsize,
        vqvae.imsize).contiguous()

    h = vqvae.encode(s, flatten=False)

    plt.figure(figsize=(30, 7))

    ########################################
    dists = np.zeros((horizon, horizon), dtype=np.float32)
    for t_a in range(horizon):
        h_a = h[t_a]
        for t_b in range(horizon):
            h_b = h[t_b]

            dists[t_a, t_b] = np.linalg.norm(
                ptu.get_numpy(torch.flatten(h_a)) -
                ptu.get_numpy(torch.flatten(h_b)),
                axis=-1)

    plt.subplot(1, 3, 1)
    df_cm = pd.DataFrame(
        dists,
        index=['t%02d' % t for t in range(horizon)],
        columns=['t%02d' % t for t in range(horizon)])
    sn.heatmap(df_cm, annot=True)
    plt.title('L2 Dist')

    ########################################
    logps = np.zeros((horizon, horizon), dtype=np.float32)
    for t_a in range(horizon):
        h_a = h[t_a]
        for t_b in range(horizon):
            h_b = h[t_b]
            z_ab, _ = affordance.encode(
                torch.unsqueeze(h_a, 0),
                cond=torch.unsqueeze(h_b, 0))
            z_ab = ptu.get_numpy(z_ab)
            print(t_a, t_b, z_ab, 0.5 * np.sum(np.square(z_ab), axis=-1))
            logps[t_a, t_b] = 0.5 * np.sum(np.square(z_ab), axis=-1)

    plt.subplot(1, 3, 2)
    df_cm = pd.DataFrame(
        logps,
        index=['t%02d' % t for t in range(horizon)],
        columns=['t%02d' % t for t in range(horizon)])
    sn.heatmap(df_cm, annot=True)
    plt.title('z^2')

    ########################################
    preds = np.zeros((horizon, horizon), dtype=np.float32)
    for t_a in range(horizon):
        h_a = h[t_a]
        for t_b in range(horizon):
            h_b = h[t_b]
            logit_ab = classifier(
                torch.unsqueeze(h_a, 0),
                torch.unsqueeze(h_b, 0))
            pred_ab = torch.sigmoid(logit_ab)
            preds[t_a, t_b] = ptu.get_numpy(pred_ab)

    plt.subplot(1, 3, 3)
    df_cm = pd.DataFrame(
        preds,
        index=['t%02d' % t for t in range(horizon)],
        columns=['t%02d' % t for t in range(horizon)])
    sn.heatmap(df_cm, annot=True)
    plt.title('classifier')


def evaluate_state_transitions(data):
    horizon = data.shape[0]
    data = ptu.from_numpy(data)
    s = data

    plt.figure(figsize=(10, 7))

    ########################################
    dists = np.zeros((horizon, horizon), dtype=np.float32)
    for t_a in range(horizon):
        s_a = s[t_a]
        for t_b in range(horizon):
            s_b = s[t_b]

            dists[t_a, t_b] = np.linalg.norm(
                ptu.get_numpy(torch.flatten(s_a)) -
                ptu.get_numpy(torch.flatten(s_b)),
                axis=-1)

    plt.subplot(1, 1, 1)
    df_cm = pd.DataFrame(
        dists,
        index=['t%02d' % t for t in range(horizon)],
        columns=['t%02d' % t for t in range(horizon)])
    sn.heatmap(df_cm, annot=True)
    plt.title('L2 Dist (State Space)')


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
    data = datasets['test'].data[FLAGS.id, ::FLAGS.delta_t]

    logging.info('Plotting the sequence...')
    plot_sequence(data)

    logging.info('Plotting the transitions...')
    evaluate_transitions(model, data)
    evaluate_state_transitions(data)

    plt.show()


if __name__ == '__main__':
    flags.mark_flag_as_required('root_dir')
    app.run(main)
