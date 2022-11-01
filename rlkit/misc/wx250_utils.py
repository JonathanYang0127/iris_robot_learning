import pickle5 as pickle
import numpy as np
import random
import torch
import rlkit.torch.pytorch_util as ptu
import copy

class DummyEnv:

    def __init__(self, image_size=128, task_embedding_dim=0):
        from gym import spaces
        self.image_size = image_size
        self.action_space = spaces.Box(
                np.asarray([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
                np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
                dtype=np.float32)
        self.observation_space = spaces.dict.Dict({
            "image": spaces.Box(
                low=np.array([0]*self.image_size*self.image_size*3),
                high=np.array([255]*self.image_size*self.image_size*3),
                dtype=np.uint8),
            "desired_pose": spaces.Box(-np.full(7, np.inf), np.full(7, np.inf),
                                dtype=np.float64),
            "current_pose": spaces.Box(-np.full(7, np.inf), np.full(7, np.inf),
                                       dtype=np.float64),
            "joint_positions": spaces.Box(-np.full(6, np.inf), np.full(6, np.inf),
                                       dtype=np.float64),
            "joint_velocities": spaces.Box(-np.full(6, np.inf), np.full(6, np.inf),
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


def relabel_data(data):
    import torch
    import torch.nn as nn
    checkpoint = '/iris/u/jyang27/dev/iris_robots/widowx_scripts/checkpoints_cdp_normalized/output_99900000.pt'
    x_mean, x_std = np.array([-1.18264896e-04, -6.12960210e-05, -1.21611653e-04,  1.87867629e-03,
        5.39370408e-04, -3.54309147e-03, -3.69889551e-03,  2.13250757e-01,
       -3.63479105e-03,  1.13428987e-01,  6.58944349e-02, -1.45073349e+00,
       -8.18877650e-02,  7.33097816e-01,  2.03012627e-01,  3.74388412e-03,
        1.15490606e-01,  3.40435020e-01, -1.35268215e+00,  1.40643964e-01,
        4.16875775e-01, -6.71489714e-03, -1.70911032e-02,  2.05775884e-02,
        1.57008383e-03, -2.14300624e-02, -5.31297725e-03,  2.13365561e-01,
       -3.57174906e-03,  1.13556635e-01,  6.21405608e-02, -1.45127918e+00,
       -8.38938782e-02,  7.36793169e-01,  2.03269970e-01,  3.71451323e-03,
        1.15503602e-01,  3.36357247e-01, -1.35370831e+00,  1.38559764e-01,
        4.11598181e-01, -6.75485685e-03, -1.63784883e-02,  2.06558428e-02,
        1.54344401e-03, -2.17131104e-02, -5.62433005e-03,  2.13132492e-01,
       -3.69608707e-03,  1.13307375e-01,  6.77731112e-02, -1.45019412e+00,
       -8.54308565e-02,  7.29398920e-01]), np.array([1.23669478e-03, 1.48516601e-03, 2.17702979e-03, 4.63521174e-01,
       4.29992223e-03, 6.21580020e-01, 1.63691752e-02, 4.11308446e-02,
       7.49869543e-02, 6.17511719e-02, 1.85217339e+00, 7.81366552e-02,
       1.92467076e+00, 3.41713682e-01, 5.56665760e-02, 8.66267518e-02,
       8.97607208e-02, 2.23733857e+00, 1.24168990e-01, 1.46854508e+00,
       4.92372058e-01, 1.11235653e-01, 1.31411771e-01, 8.91964575e-02,
       7.11472804e-02, 1.54575382e-01, 2.04479587e-01, 4.10605037e-02,
       7.49626141e-02, 6.18620879e-02, 1.84180877e+00, 7.79690100e-02,
       1.93640908e+00, 3.40930207e-01, 5.54956107e-02, 8.65781454e-02,
       8.97131176e-02, 2.23066613e+00, 1.24278704e-01, 1.48389012e+00,
       4.91418124e-01, 1.10665713e-01, 1.31070933e-01, 8.88201365e-02,
       7.08571317e-02, 1.54311705e-01, 2.03996488e-01, 4.12024466e-02,
       7.50107412e-02, 6.16437085e-02, 1.86230982e+00, 7.83045626e-02,
       1.91304946e+00, 3.42430888e-01])
    y_mean, y_std = np.array([-0.01049738,  0.00741228,  0.00206169,  0.28555748,  0.09901281,
        0.2241386 , -0.31105257]), np.array([0.04346715, 0.05537778, 0.08476445, 2.6374713 , 0.1143113 ,
       2.09822954, 0.82471913])

    model = nn.Sequential(
            nn.Linear(54, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 7)
            ).cuda()
    model.load_state_dict(torch.load(checkpoint))
    for i in range(len(data)):
        path = data[i]
        #TODO: Handle edge case when there is no previous observation at the beginning
        for t in range(1, len(path['observations']) - 1):
            current_pose = path['observations'][t]['current_pose']
            next_achieved_pose = path['observations'][t + 1]['current_pose']
            adp = (next_achieved_pose - current_pose).tolist()
            for k in range(0, 2):
                adp += copy.deepcopy(path['observations'][t-k]['current_pose']).tolist()
                adp += copy.deepcopy(path['observations'][t-k]['desired_pose']).tolist()
                adp += copy.deepcopy(path['observations'][t-k]['joint_velocities']).tolist()
            adp += copy.deepcopy(next_achieved_pose).tolist()
            adp = np.array(adp)
            adp = torch.Tensor((adp - x_mean) / x_std).cuda()
            data[i]['actions'][t] = model(adp).detach().cpu().numpy() * y_std + y_mean
    return data


def load_data(data_object):
    if isinstance(data_object, str):
        try:
            if '.pkl' in data_object:
                with open(data_object, 'rb') as handle:
                    data = pickle.load(handle)
            elif '.npy' in data_object:
                with open(data_object, 'rb') as handle:
                    data = np.load(handle, allow_pickle=True)

            #Create next_observations if it doesn't exist
            for i in range(len(data)):
                if 'next_observations' not in data[i].keys():
                    pathlength = len(data[i]['observations'])
                    data[i]['next_observations'] = copy.deepcopy(data[i]['observations'])
                    for j in range(pathlength - 1):
                        data[i]['next_observations'][j] = copy.deepcopy(data[i]['observations'][j + 1])
                    #hack that sets next observation of the last timestep to the observation
                    #a better way to handle this might be to delete the last timestep
                    data[i]['next_observations'][pathlength - 1] = \
                        copy.deepcopy(data[i]['observations'][pathlength - 1])

            # Hack to create transfer rgb data from "images" to "image"
            if 'image' not in data[0]['observations'][0].keys():
                for i in range(len(data)):
                    pathlength = len(data[i]['observations'])
                    for j in range(pathlength):
                        data[i]['observations'][j]['image'] = \
                            copy.deepcopy(data[i]['observations'][j]['images'][0]['array'])
                        data[i]['next_observations'][j]['image'] = \
                            copy.deepcopy(data[i]['next_observations'][j]['images'][0]['array'])

            #Zero rewards and terminals if it doesn't exist
            if 'rewards' not in data[0].keys():
                for i in range(len(data)):
                    data[i]['rewards'] = np.zeros((len(data[i]['observations']), 1))
            if 'terminals' not in data[0].keys():
                for i in range(len(data)):
                    data[i]['terminals'] = np.zeros((len(data[i]['observations']), 1))

            #Relabel data
            data = relabel_data(data)

            #Throw away final transitions (for consistent relabelling and next observation generation)
            for i in range(len(data)):
                for k in ['observations', 'next_observations', 'actions', 'rewards', 'terminals']:
                    np.delete(data[i][k], -1)
        except:
            print("Failed to open {}".format(data_object))
        return data
    elif isinstance(data_object, list) or isinstance(data_object, np.ndarray):
        return data_object
    else:
        raise NotImplementedError

def process_image(image):
    ''' ObsDictReplayBuffer wants flattened (channel, height, width) images float32'''
    if image.dtype == np.uint8:
        image =  image.astype(np.float32) / 255.0
    if len(image.shape) == 3 and image.shape[0] != 3 and image.shape[2] == 3:
        image = np.transpose(image, (2, 0, 1))
        image = image.flatten()
    return image

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
