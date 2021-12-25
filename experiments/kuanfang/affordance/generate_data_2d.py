import os
import random
from absl import app
from absl import flags
from absl import logging  # NOQA

import numpy as np

from lib.envs.toy_2d import Toy2DEnv  # NOQA
from lib.envs.brownian_env import BrownianEnv  # NOQA

flags.DEFINE_string(
    'output_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
    'output directory for writing logs/summaries/checkpoints.')
flags.DEFINE_integer('seed', None, 'Random seed.')
flags.DEFINE_integer('num_samples', 10000, 'Size of the dataset.')
flags.DEFINE_float('ratio', 0.9, 'Ratio of the training data')
flags.DEFINE_string('mode', 'pos', 'Mode of the environment.')

FLAGS = flags.FLAGS


def generate_data(output_dir, num_samples, ratio, mode):
    output_dir = os.path.expanduser(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, 'data.npy')
    logging.info('The data will be saved to %s.', output_path)

    data = []

    # env = Toy2DEnv(
    #     num_bodies=3,
    #     mode=mode)
    env = BrownianEnv(
        num_bodies=3)

    logging.info('Start generating %d samples...', num_samples)
    for sample_id in range(num_samples):
        transition = env.sample_transition()
        # env.render_transition(transition)  # TODO
        data.append(transition)

        if (sample_id + 1) % 1000 == 0:
            logging.info('%d / %d', sample_id + 1, num_samples)

    num_train = int(num_samples * ratio)
    data = np.stack(data, axis=0)
    train_data = data[:num_train]
    test_data = data[num_train:]
    # logging.info('Split into train (%d) and test (%d).',
    #              num_train, num_samples - num_train)
    logging.info('Train: %r, Test: %r', train_data.shape, test_data.shape)
    data_dict = {
        'train': train_data,
        'test': test_data,
    }

    np.save(output_path, data_dict)
    logging.info('Saved the data to %s.', output_path)


def main(_):
    if FLAGS.seed is not None:
        random.seed(FLAGS.seed)
        np.random.seed(FLAGS.seed)

    generate_data(output_dir=FLAGS.output_dir,
                  num_samples=FLAGS.num_samples,
                  ratio=FLAGS.ratio,
                  mode=FLAGS.mode)


if __name__ == '__main__':
    flags.mark_flag_as_required('output_dir')
    app.run(main)
