import pickle5 as pickle
import numpy as np
import random


class DummyEnv:

    def __init__(self, image_size, use_wrist=False, num_tasks=0):
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
        if num_tasks > 0:
            self.observation_space.spaces.update({
                'task': spaces.Box(
                     low=np.array([0] * num_tasks),
                     high=np.array([1] * num_tasks),
                 )})

    def step(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


def load_data(data_object):
    if isinstance(data_object, str):
        with open(data_object, 'rb') as handle:
            data = pickle.load(handle)
        return data
    elif isinstance(data_object, list):
        return data_object
    else:
        raise NotImplementedError

def add_data_to_buffer_real_robot(data, replay_buffer, validation_replay_buffer=None,
                       validation_fraction=0.8, num_trajs_limit=None):
    assert validation_fraction >= 0.0
    assert validation_fraction < 1.0

    paths = load_data(data)

    if num_trajs_limit is not None:
        assert num_trajs_limit <= len(paths)
        # Shuffle trajectories before truncating dataset
        random.shuffle(paths)
        paths = paths[:num_trajs_limit]

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

    print("replay_buffer._size", replay_buffer._size)


# TODO: Add validation buffers
def add_multitask_data_to_singletask_buffer_real_robot(data_paths, replay_buffer):

    assert isinstance(data_paths, dict)
    assert 'task' in replay_buffer.observation_keys

    for task, data_path in data_paths.items():
        paths = load_data(data_path)
        for path in paths:
            for i in range(len(path['observations'])):
                path['observations'][i]['task'] = np.array([0] * len(data_paths.keys()))
                path['observations'][i]['task'][task] = 1
                path['next_observations'][i]['task'] = np.array([0] * len(data_paths.keys()))
                path['next_observations'][i]['task'][task] = 1
        add_data_to_buffer_real_robot(paths, replay_buffer)


def add_multitask_data_to_multitask_buffer_real_robot(data_paths, multitask_replay_buffer):

    assert isinstance(data_paths, dict)

    for task, data_path in data_paths.items():
        add_data_to_buffer_real_robot(data_path, multitask_replay_buffer.task_buffers[task])

