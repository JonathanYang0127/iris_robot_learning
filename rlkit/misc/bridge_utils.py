import pickle5 as pickle
import numpy as np
import random
import torch
import rlkit.torch.pytorch_util as ptu


class DummyEnv:

    def __init__(self, image_size, use_wrist=False, task_embedding_dim=0):
        from gym import spaces
        self.image_size = image_size
        self.action_space = spaces.Box(
	    np.asarray([-0.05, -0.05, -0.05, -0.05, -0.05, -0.05, -1.0]),
	    np.asarray([0.05, 0.05, 0.05, 0.05, 0.05, 0.0, 1.0]),
	    dtype=np.float32)
        self.observation_space = spaces.dict.Dict({
            "image": spaces.Box(
                low=np.array([0]*self.image_size*self.image_size*3),
                high=np.array([255]*self.image_size*self.image_size*3),
                dtype=np.uint8),
            "state": spaces.Box(-np.full(8, np.inf), np.full(8, np.inf),
                                dtype=np.float64),
        })
        if task_embedding_dim > 0:
            self.observation_space.spaces.update({
                'task_embedding': spaces.Box(
                     low=np.array([-10] * task_embedding_dim),
                     high=np.array([10] * task_embedding_dim),
                 )})

    def step(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


def load_data(data_object):
    if isinstance(data_object, str):
        try:
            if 'pkl' in data_object:
                with open(data_object, 'rb') as handle:
                    data = pickle.load(handle)
            elif 'npy' in data_object:
                data = np.load(data_object, allow_pickle=True)
           
            for i in range(len(data)):
                #Convert ndarray observations into obs dict
                #We are taking the 0th image of the 3 for the robot view
                temp1, temp2 = data[i]['observations'], data[i]['next_observations']
                data[i]['observations'], data[i]['next_observations'] = list(), list()
                for j in range(len(temp1)):
                    data[i]['observations'].append({'image': temp1[j][0]})
                    data[i]['next_observations'].append({'image': temp2[j][0]})

                #Reshape terminals
                data[i]['terminals'] = np.reshape(data[i]['terminals'], (-1, 1))
                #Reshape rewards
                data[i]['rewards'] = np.reshape(data[i]['rewards'], (-1, 1))
        except:
            print("Failed to open {}".format(data_object))
        return data
    elif isinstance(data_object, list) or isinstance(data_object, np.ndarray):
        return data_object
    else:
        raise NotImplementedError

def process_image(image):
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    elif image.dtype == np.float32:
        return image
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
            for i in range(len(path['observations'])):
                path['observations'][i]['image'] = process_image(path['observations'][i]['image'])
                path['next_observations'][i]['image'] = process_image(path['next_observations'][i]['image'])
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
def add_multitask_data_to_singletask_buffer_real_robot(data_paths, replay_buffer, task_encoder=None, embedding_mode='single',
        encoder_type='image', num_tasks=None):

    assert isinstance(data_paths, dict)
    # assert 'task_embedding' in replay_buffer.observation_keys
    assert embedding_mode in ('one-hot', 'single', 'batch', 'None')
    assert encoder_type in ('image', 'trajectory')

    use_task_encoder = task_encoder is not None
    if num_tasks is None:
        num_tasks = len(data_paths.keys())

    if use_task_encoder:
        from rlkit.data_management.multitask_replay_buffer import ObsDictMultiTaskReplayBuffer
        buffer_args = (
            replay_buffer.max_size,
            replay_buffer.env,
            np.arange(num_tasks))
        buffer_kwargs = dict(
            use_next_obs_in_context=False,
            sparse_rewards=False,
            observation_keys=['image']
        )
        if encoder_type == 'image':
            encoder_buffer = ObsDictMultiTaskReplayBuffer(*buffer_args, **buffer_kwargs)
            add_reward_filtered_data_to_buffers_multitask(data_paths, ['image'],
                (encoder_buffer, lambda r: r > 0))
            sample_func = encoder_buffer.sample_batch
        elif encoder_type == 'trajectory':
            buffer_kwargs['observation_keys'] = ['image', 'state']
            encoder_buffer = ObsDictMultiTaskReplayBuffer(*buffer_args, **buffer_kwargs)
            add_reward_filtered_trajectories_to_buffers_multitask(data_paths, ['image'],
                (encoder_buffer, lambda t: np.sum(t) > 0))
            sample_func = encoder_buffer.sample_batch_of_trajectories

    if use_task_encoder and embedding_mode == 'single':
        encoder_batch = sample_func(data_paths.keys(), 100)
        z, mu, logvar = task_encoder.forward(ptu.from_numpy(encoder_batch['observations']))
        mu = torch.mean(mu.view(len(data_paths.keys()), 100, -1), dim=1)
    for task_idx, data_path in data_paths.items():
        paths = load_data(data_path)
        for path in paths:
            if use_task_encoder and embedding_mode=='batch':
                encoder_batch = sample_func([task_idx], len(path['observations']))
                z, mu, logvar = task_encoder.forward(ptu.from_numpy(encoder_batch['observations']))
            for i in range(len(path['observations'])):
                if use_task_encoder and embedding_mode == 'single':
                    task_embedding_mean = ptu.get_numpy(mu[task_idx])
                    path['observations'][i]['task_embedding'] = task_embedding_mean
                    path['next_observations'][i]['task_embedding'] = task_embedding_mean
                elif use_task_encoder and embedding_mode == 'batch':
                    task_embedding_mean = ptu.get_numpy(mu[i])
                    path['observations'][i]['task_embedding'] = task_embedding_mean
                    path['next_observations'][i]['task_embedding'] = task_embedding_mean
                elif embedding_mode == 'one-hot':
                    path['observations'][i]['task_embedding'] = np.array([0] * num_tasks)
                    path['observations'][i]['task_embedding'][task_idx] = 1
                    path['next_observations'][i]['task_embedding'] = np.array([0] * num_tasks)
                    path['next_observations'][i]['task_embedding'][task_idx] = 1
                elif embedding_mode == 'None':
                    pass
                else:
                    raise NotImplementedError

        add_data_to_buffer_real_robot(paths, replay_buffer)

def add_multitask_data_to_multitask_buffer_real_robot(data_paths, multitask_replay_buffer, task_encoder=None, embedding_mode='None',
        encoder_type='image', num_tasks=None):

    assert isinstance(data_paths, dict)

    if num_tasks is None:
        num_tasks = len(data_paths.keys())
    for task, data_path in data_paths.items():
        add_multitask_data_to_singletask_buffer_real_robot({task: data_path}, multitask_replay_buffer.task_buffers[task],
            task_encoder=task_encoder, embedding_mode=embedding_mode, encoder_type=encoder_type, num_tasks=num_tasks)

def add_reward_filtered_data_to_buffers_multitask(data_paths, observation_keys, *args):
    for arg in args:
        assert len(arg) == 2

    for task_idx, data_path in data_paths.items():
        data_paths[task_idx] = load_data(data_path)
        data = data_paths[task_idx]

        for j in range(len(data)):
            path_len = len(data[j]['actions'])
            path = data[j]
            for i in range(path_len):
                for arg in args:
                    if arg[1](path['rewards'][i]):
                        path['observations'][i]['image'] = process_image(path['observations'][i]['image'])
                        path['next_observations'][i]['image'] = process_image(path['next_observations'][i]['image'])
                        arg[0].add_sample(task_idx,
                            path['observations'][i], path['actions'][i], path['rewards'][i],
                            path['terminals'][i], path['next_observations'][i])

def add_reward_filtered_trajectories_to_buffers_multitask(data_paths, observation_keys, *args):
    for arg in args:
        assert len(arg) == 2

    for task_idx, data_path in data_paths.items():
        data_paths[task_idx] = load_data(data_path)
        data = data_paths[task_idx]

        for j in range(len(data)):
            path_len = len(data[j]['actions'])
            path = data[j]
            for arg in args:
                if arg[1](path['rewards']):
                    for i in range(path_len):
                        path['observations'][i]['image'] = process_image(path['observations'][i]['image'])
                        path['next_observations'][i]['image'] = process_image(path['next_observations'][i]['image'])
                        arg[0].add_sample(task_idx,
                            path['observations'][i], path['actions'][i], path['rewards'][i],
                            path['terminals'][i], path['next_observations'][i])