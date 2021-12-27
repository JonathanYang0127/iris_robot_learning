import os
from absl import app
from absl import flags
from absl import logging  # NOQA

import numpy as np

flags.DEFINE_string(
    'input', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'), '...')
flags.DEFINE_string(
    'output', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'), '...')
flags.DEFINE_integer('num_train', None, '...')
flags.DEFINE_integer('num_test', None, '...')

FLAGS = flags.FLAGS


def read_dataset(file_path):
    data = np.load(file_path, allow_pickle=True)
    data = data.item()

    data = data['observations']

    return data


def convert_val_dataset(
        input_path,
        output_dir,
        num_train,
        num_test,
):
    output_dir = os.path.expanduser(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, 'data.npy')

    data = np.load(input_path, allow_pickle=True)
    data = data.item()

    train_data = data['train'][:num_train]
    test_data = data['test'][:num_test]

    logging.info('Train: %r, Test: %r', train_data.shape, test_data.shape)
    data_dict = {
        'train': train_data,
        'test': test_data,
    }

    np.save(output_path, data_dict)
    logging.info('Saved the data to %s.', output_path)


def main(_):
    convert_val_dataset(
        input_path=FLAGS.input,
        output_dir=FLAGS.output,
        num_train=FLAGS.num_train,
        num_test=FLAGS.num_test,
    )


if __name__ == '__main__':
    app.run(main)
