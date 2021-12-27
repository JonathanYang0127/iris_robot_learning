import os
from absl import app
from absl import flags

import numpy as np


flags.DEFINE_string('data', None, 'Path to the data.')
flags.DEFINE_integer('seed', None, 'Random seed.')

FLAGS = flags.FLAGS


def main(_):
    # print('Loading...')
    # data = np.load(
    #     os.path.join(FLAGS.data, 'data.npy'),
    #     allow_pickle=True)
    # data = data.item()
    # print('Saving train data...')
    # np.save(
    #     os.path.join(FLAGS.data, 'train_data.npy'),
    #     data['train'])
    # print('Saving test data...')
    # np.save(
    #     os.path.join(FLAGS.data, 'test_data.npy'),
    #     data['test'])

    print('Loading...')
    encoding = np.load(
        os.path.join(FLAGS.data, 'encoding.npy'),
        allow_pickle=True)
    encoding = encoding.item()
    print('Saving train encoding...')
    np.save(
        os.path.join(FLAGS.data, 'train_encoding.npy'),
        encoding['train'])
    print('Saving test encoding...')
    np.save(
        os.path.join(FLAGS.data, 'test_encoding.npy'),
        encoding['test'])


if __name__ == '__main__':
    app.run(main)
