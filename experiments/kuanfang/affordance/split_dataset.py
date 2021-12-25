import os
from absl import app
from absl import flags
from absl import logging  # NOQA

import numpy as np

flags.DEFINE_string(
    'input', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'), '...')
flags.DEFINE_string(
    'output', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'), '...')
flags.DEFINE_integer('num_samples', 100, '...')

FLAGS = flags.FLAGS


def convert_val_dataset(
        input_path,
        output_dir,
        num_samples,
):
    output_dir = os.path.expanduser(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data = np.load(input_path, allow_pickle=True)
    data = data.item()

    train_data = data['train']
    test_data = data['test']

    num_train_partitions = int(train_data.shape[0] / num_samples)
    num_test_partitions = int(test_data.shape[0] / num_samples)

    for i in range(num_train_partitions):
        train_data_i = train_data[i * num_samples:(i+1) * num_samples]
        # data_dict = {
        #     'train': train_data_i,
        # }
        data_dict = train_data_i
        output_path = os.path.join(output_dir, 'data_train%02d.npy' % (i))
        logging.info('Saving the train data %d / %d...',
                     i, num_train_partitions)
        np.save(output_path, data_dict)

    for i in range(num_test_partitions):
        test_data_i = test_data[i * num_samples:(i+1) * num_samples]
        # data_dict = {
        #     'test': test_data_i,
        # }
        data_dict = test_data_i
        output_path = os.path.join(output_dir, 'data_test%02d.npy' % (i))
        logging.info('Saving the test data %d / %d...',
                     i, num_test_partitions)
        np.save(output_path, data_dict)


def main(_):
    convert_val_dataset(
        input_path=FLAGS.input,
        output_dir=FLAGS.output,
        num_samples=FLAGS.num_samples,
    )


if __name__ == '__main__':
    app.run(main)
