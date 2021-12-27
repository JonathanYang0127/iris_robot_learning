import os
import random
from absl import app
from absl import flags
from absl import logging  # NOQA

import numpy as np
import matplotrlkit.experimental.kuanfang.pyplot as plt
import pickle  # NOQA
import torch

from rlkit.torch import pytorch_util as ptu

from rlkit.experimental.kuanfang.vae import vae_datasets
from rlkit.experimental.kuanfang.utils import io_util
from rlkit.experimental.kuanfang.utils import device_util


flags.DEFINE_string('root_dir', None, 'Root directory.')
flags.DEFINE_string('data_dir', None, 'Path to the data.')
flags.DEFINE_integer('seed', None, 'Random seed.')

FLAGS = flags.FLAGS


def load_model(root_dir):
    vqvae_path = os.path.join(root_dir, 'vqvae.pt')

    vqvae = torch.load(vqvae_path)

    return {
        'vqvae': vqvae,
    }


def load_dataset(dataset_path):
    dataset_ctor = vae_datasets.VaeDataset

    train_dataset = dataset_ctor(
        dataset_path, train=True, transform=None, crop=[48, 48])
    test_dataset = dataset_ctor(
        dataset_path, train=False, transform=None, crop=[48, 48])

    return {
        'train': train_dataset,
        'test': test_dataset,
    }


def encode_dataset(model, data, batch_size=1024, mode='zq'):
    vqvae = model['vqvae']

    num_seqs = data.shape[0]
    num_steps = data.shape[1]
    num_samples = num_seqs * num_steps

    data = data.reshape([-1, data.shape[-3], data.shape[-2], data.shape[-1]])

    # data = ptu.from_numpy(data)
    # _data = data.view(-1, data.shape[-3], data.shape[-2], data.shape[-1])

    num_batches = int(np.ceil(float(num_samples) / float(batch_size)))
    encodings = []
    for i in range(num_batches):
        start = i * batch_size
        end = (i + 1) * batch_size
        end = min(end, num_samples)

        batch = data[start:end]
        batch = ptu.from_numpy(batch)
        batch = batch.view(
            -1, batch.shape[-3], batch.shape[-2], batch.shape[-1])

        encoding_i = vqvae.encode(batch, mode=mode, flatten=False)
        encoding_i = ptu.get_numpy(encoding_i)
        encodings.append(encoding_i)
        logging.info('Finished encoding the data %d / %d.'
                     % (end, num_samples))

    encodings = np.concatenate(encodings, axis=0)
    logging.info('encodings.shape: %r', encodings.shape)
    if mode == 'zq':
        encodings = np.reshape(
            encodings,
            (num_seqs,
             num_steps,
             encodings.shape[-3],
             encodings.shape[-2],
             encodings.shape[-1])
        )
    elif mode == 'zi':
        encodings = np.reshape(
            encodings,
            (num_seqs,
             num_steps,
             encodings.shape[-2],
             encodings.shape[-1])
        )
    else:
        raise NotImplementedError
    return encodings


def main(_):
    if FLAGS.seed is not None:
        random.seed(FLAGS.seed)
        np.random.seed(FLAGS.seed)
        torch.manual_seed(FLAGS.seed)

    device_util.set_device(True)

    logging.info('Loading the model from %s ...' % (FLAGS.root_dir))
    model = io_util.load_model(FLAGS.root_dir)

    logging.info('Loading the dataset from %s ...' % (FLAGS.data_dir))
    datasets = io_util.load_datasets(FLAGS.data_dir)

    logging.info('Encoding the dataset...')
    test_data = datasets['test'].data
    train_data = datasets['train'].data
    test_encoding = encode_dataset(model, test_data)
    train_encoding = encode_dataset(model, train_data)

    logging.info('Train: %r, Test: %r', train_data.shape, test_data.shape)

    output_path = os.path.join(FLAGS.root_dir, 'train_encoding.npy')
    np.save(output_path, train_encoding)

    output_path = os.path.join(FLAGS.root_dir, 'test_encoding.npy')
    np.save(output_path, test_encoding)

    logging.info('Saved the data to %s.', output_path)

    plt.show()


if __name__ == '__main__':
    flags.mark_flag_as_required('root_dir')
    app.run(main)
