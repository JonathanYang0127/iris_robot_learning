import os
from absl import app
from absl import flags
from absl import logging  # NOQA

import numpy as np

from lib.envs.toy_2d import Toy2DEnv  # NOQA
from lib.envs.brownian_env import BrownianEnv  # NOQA
from lib.envs.val import ValEnv  # NOQA
from lib.vae import vae_datasets

flags.DEFINE_string(
    'dataset', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
    'dataset directory.')
flags.DEFINE_string('mode', 'pos', 'Mode of the environment.')
flags.DEFINE_boolean('training', True, 'Whether it is training.')
flags.DEFINE_boolean('random', True, 'Whether it is randomly sampled.')

FLAGS = flags.FLAGS


def visualize_dataset(
        dataset_path,
        mode,
        training,
        random,
        is_val):

    if is_val:
        # dataset = vae_datasets.ValDataset(
        #     dataset_path, train=None, transform=None, preprocess_image=False)
        dataset = vae_datasets.ValTransitionDataset(
            dataset_path, train=None, transform=None, preprocess_image=False)
    elif training:
        dataset = vae_datasets.VaeDataset(
            dataset_path, train=True, transform=None, preprocess_image=False)
    else:
        dataset = vae_datasets.VaeDataset(
            dataset_path, train=False, transform=None, preprocess_image=False)

    # TODO
    # if is_val:
    #     env = ValEnv()
    # else:
    #     env = Toy2DEnv(mode=mode)

    env = BrownianEnv(num_bodies=3)

    for i in range(len(dataset)):
        if random:
            sample_id = i
        else:
            sample_id = int(np.random.choice(len(dataset)))

        logging.info('Render sample %d / %d.', sample_id, len(dataset))
        transition = dataset[sample_id]
        print(transition.shape)

        env.render_transition(transition)

    logging.info('Saved the data to %s.', dataset_path)


def main(_):
    visualize_dataset(
        dataset_path=FLAGS.dataset,
        mode=FLAGS.mode,
        training=FLAGS.training,
        random=FLAGS.random,
        is_val=0)  # TODO


if __name__ == '__main__':
    app.run(main)
