import pickle5 as pickle
import numpy as np
import random


class DummyEnv:

    def __init__(self, image_size, use_wrist=False):
        from gym import spaces
        self.image_size = image_size
        if not use_wrist:
            self.action_space = spaces.Box(
                np.asarray([-0.05, -0.05, -0.05, -1.0]),
                np.asarray([0.05, 0.05, 0.05, 1.0]),
                dtype=np.float32)
        else:
            self.action_space = spaces.Box(
		np.asarray([-0.05, -0.05, -0.05, -1.0, -1.0]),
		np.asarray([0.05, 0.05, 0.05, 1.0, 1.0]),
		dtype=np.float32)
        self.observation_space = spaces.dict.Dict({
            "image": spaces.Box(
                low=np.array([0]*self.image_size*self.image_size*3),
                high=np.array([255]*self.image_size*self.image_size*3),
                dtype=np.uint8),
            "state": spaces.Box(-np.full(8, np.inf), np.full(8, np.inf),
                                dtype=np.float64),
        })

    def step(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


def add_data_to_buffer_real_robot(data_path, replay_buffer, validation_replay_buffer=None,
                       validation_fraction=0.8):
    with open(data_path, 'rb') as handle:
        paths = pickle.load(handle)

    assert validation_fraction >= 0.0
    assert validation_fraction < 1.0

    if validation_replay_buffer is None:
        for path in paths:
            replay_buffer.add_path(path)
    else:
        num_train = int(validation_fraction*len(paths))
        random.shuffle(paths)
        train_paths = paths[:num_train]
        val_paths = paths[:num_train]

        for path in train_paths:
            replay_buffer.add_path(path)

        for path in val_paths:
            validation_replay_buffer.add_path(path)

