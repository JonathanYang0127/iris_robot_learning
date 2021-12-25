import os
from absl import app
from absl import flags
from absl import logging  # NOQA

import numpy as np

flags.DEFINE_string(
    'train', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'), '...')
flags.DEFINE_string(
    'test', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'), '...')
flags.DEFINE_string(
    'output', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'), '...')

FLAGS = flags.FLAGS


def read_val_dataset(file_path):
    data = np.load(file_path, allow_pickle=True)
    data = data.item()

    data = data['observations']

    num_samples = data.shape[0]
    num_steps = data.shape[1]
    data = np.reshape(data, [num_samples, num_steps, 3, 48, 48])
    data = np.transpose(data, [0, 1, 4, 3, 2])  # TODO
    return data


def convert_val_dataset(
        train_path,
        test_path,
        output_dir,
):
    output_dir = os.path.expanduser(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, 'data.npy')

    train_data = read_val_dataset(train_path)
    test_data = read_val_dataset(test_path)

    logging.info('Train: %r, Test: %r', train_data.shape, test_data.shape)
    data_dict = {
        'train': train_data,
        'test': test_data,
    }

    np.save(output_path, data_dict)
    logging.info('Saved the data to %s.', output_path)


def main(_):
    convert_val_dataset(
        train_path=FLAGS.train,
        test_path=FLAGS.test,
        output_dir=FLAGS.output,
    )


if __name__ == '__main__':
    app.run(main)
