import pickle5 as pickle
import numpy as np
import random
import torch
import rlkit.torch.pytorch_util as ptu
from scipy.spatial.transform import Rotation as R
import copy

_STACK_FRAMES = False
_DOWNSAMPLE_IMAGE = False
_MIXUP = False
_ACTION_RELABELLING = None
_ACTION_ALIGNMENT = False

def configure_dataloader_params(variant):
    global _STACK_FRAMES, _DOWNSAMPLE_IMAGE, _MIXUP, _ACTION_RELABELLING, _ACTION_ALIGNMENT
    _STACK_FRAMES = variant['dataloader_params']['stack_frames']
    _DOWNSAMPLE_IMAGE = variant['dataloader_params']['downsample_image']
    _MIXUP = variant['dataloader_params']['mixup']
    _ACTION_RELABELLING = variant['dataloader_params']['action_relabelling']
    _ACTION_ALIGNMENT = variant['dataloader_params']['align_actions']

class DummyEnv:

    def __init__(self, image_size=128, task_embedding_dim=0):
        from gym import spaces
        self.image_size = image_size
        self.action_space = spaces.Box(
                np.asarray([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
                np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
                dtype=np.float32)
        self.observation_space = spaces.dict.Dict({
            "previous_image": spaces.Box(
                low=np.array([0]*self.image_size*self.image_size*3),
                high=np.array([255]*self.image_size*self.image_size*3),
                dtype=np.uint8),
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
            "mixup_distance": spaces.Box(-np.full(7, np.inf), np.full(7, np.inf),
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


def angle_diff(target, source):
    result = R.from_euler('xyz', target) * R.from_euler('xyz', source).inv()
    return result.as_euler('xyz')


def pose_diff(target, source):
    diff = np.zeros(len(target))
    diff[:3] = target[:3] - source[:3]
    diff[3:6] = angle_diff(target[3:6], source[3:6])
    diff[6] = target[6] - source[6]
    return diff


def relabel_actions_linear(data):
    from sklearn import linear_model
    from sklearn.metrics import mean_squared_error
    import pickle

    PATH = '/iris/u/jyang27/dev/iris_robots/widowx_scripts/linear_cdp_model.pkl'
    with open(PATH, 'rb') as f:
        model = pickle.load(f)
    for i in range(len(data)):
        path = data[i]
        #TODO: Handle edge case when there is no previous observation at the beginning
        for t in range(1, len(path['observations']) - 1):
            current_pose = path['observations'][t]['current_pose']
            next_achieved_pose = path['observations'][t + 1]['current_pose']
            adp = pose_diff(next_achieved_pose, current_pose).tolist()
            for k in range(0, 2):
                adp += copy.deepcopy(path['observations'][t-k]['current_pose']).tolist()
                adp += copy.deepcopy(path['observations'][t-k]['desired_pose']).tolist()
                #adp += copy.deepcopy(path['observations'][t-k]['joint_velocities']).tolist()
            #adp += copy.deepcopy(next_achieved_pose).tolist()
            adp = np.array(adp)
            data[i]['actions'][t] = model.predict(adp)
    return data


def relabel_actions_nonlinear(data):
    import torch
    import torch.nn as nn
    checkpoint = '/iris/u/jyang27/dev/iris_robots/widowx_scripts/checkpoints_cdp_normalized/output_99900000.pt'
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
        for t in range(0, len(path['observations']) - 1):
            current_pose = path['observations'][t]['current_pose']
            next_achieved_pose = path['observations'][t + 1]['current_pose']
            adp = (next_achieved_pose - current_pose).tolist()
            for k in range(0, 2):
                index = 0 if t - k < 0 else t - k
                adp += copy.deepcopy(path['observations'][index]['current_pose']).tolist()
                adp += copy.deepcopy(path['observations'][index]['desired_pose']).tolist()
                #adp += copy.deepcopy(path['observations'][index]['joint_velocities']).tolist()
            #adp += copy.deepcopy(next_achieved_pose).tolist()
            adp = np.array(adp)
            adp = torch.Tensor(adp).cuda()
            data[i]['actions'][t] = model(adp).detach().cpu().numpy()
    return data


def relabel_achieved_actions(data):
    for i in range(len(data)):
        path = data[i]
        for t in range(0, len(path['observations']) - 1):
            current_pose = path['observations'][t]['current_pose']
            next_achieved_pose = path['observations'][t+1]['current_pose']
            data[i]['actions'][t] = 100 * pose_diff(next_achieved_pose, current_pose)
    return data


def stack_frames(data):
    image_shape = data[0]['observations'][0]['image'].shape
    image_type = data[0]['observations'][0]['image'].dtype
    channel_dim = min(image_shape)
    assert image_shape[2] == channel_dim     #height, width, channel
    
    for i in range(len(data)):
        pathlength = len(data[i]['observations'])
        for j in range(pathlength):
            previous_image = data[i]['observations'][max(0, j - 1)]['image']
            data[i]['observations'][j]['previous_image'] = previous_image
            data[i]['next_observations'][j]['previous_image'] = data[i]['observations'][j]['image']
    return data

def load_data(data_object):
    if isinstance(data_object, str):
        if '.pkl' in data_object:
            with open(data_object, 'rb') as handle:
                data = pickle.load(handle)
        elif '.npy' in data_object:
            with open(data_object, 'rb') as handle:
                data = np.load(handle, allow_pickle=True)

        #Align actions
        if _ACTION_ALIGNMENT:
            for i in range(len(data)):
                pathlength = len(data[i]['observations'])
                for j in range(pathlength):
                    if 'franka' in data_object:
                        data[i]['actions'][j][3:6] *= -1
                        data[i]['observations'][j]['current_pose'][3:6] *= -1
                        data[i]['observations'][j]['desired_pose'][3:6] *= -1

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


        if _MIXUP:
            for i in range(len(data)):
                closed_pose = None
                for j in range(len(data[i]['actions'])):
                    if data[i]['actions'][j][6] > 0.5:
                        closed_pose = data[i]['observations'][j]['current_pose']
                        break

                for j in range(len(data[i]['actions'])):
                    data[i]['observations'][j]['mixup_distance'] = \
                        100 * pose_diff(data[i]['observations'][j]['current_pose'], closed_pose)
                    data[i]['next_observations'][j]['mixup_distance'] = \
                        100 * pose_diff(data[i]['next_observations'][j]['current_pose'], closed_pose)

        #Relabel actions
        if _ACTION_RELABELLING == 'achieved':
            data = relabel_achieved_actions(data)
        elif _ACTION_RELABELLING == 'linear':
            data = relabel_actions_linear(data)
        
        #Stack frames
        if _STACK_FRAMES:
            data = stack_frames(data)

        #Throw away final transitions (for consistent relabelling and next observation generation)
        for i in range(len(data)): 
            for k in ['observations', 'next_observations', 'actions', 'rewards', 'terminals']:
                np.delete(data[i][k], -1)
        return data
    elif isinstance(data_object, list) or isinstance(data_object, np.ndarray):
        return data_object
    else:
        raise NotImplementedError

def process_image(image):
    ''' ObsDictReplayBuffer wants flattened (channel, height, width) images float32'''
    channel_dim = min(image.shape)
    image_size = max(image.shape)
    if image.dtype == np.uint8:
        image =  image.astype(np.float32) / 255.0
    if len(image.shape) == 3 and image.shape[0] != channel_dim and image.shape[2] == channel_dim:
        image = np.transpose(image, (2, 0, 1))
    if _DOWNSAMPLE_IMAGE and image_size != 64:
        #TODO: try cv2 resizing
        image = image[:,::2, ::2]
    
    return image.flatten()

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
                if _STACK_FRAMES:
                    path['observations'][i]['previous_image'] = process_image(path['observations'][i]['previous_image'])
                    path['next_observations'][i]['previous_image'] = process_image(path['next_observations'][i]['previous_image'])
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
