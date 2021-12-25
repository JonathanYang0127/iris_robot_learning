import os  # NOQA
from absl import logging  # NOQA

import numpy as np

from torch.utils.data import Dataset


class VaeDataset(Dataset):

    def __init__(self,
                 data_path,
                 encoding_path=None,
                 train=None,
                 vqvae=None,
                 transform=None,
                 crop=None,
                 preprocess_image=True,
                 channel_first=True,
                 is_val_format=False,
                 ):

        # Load the data and the encoding.
        logging.info('Loading the dataset from %s...', data_path)
        data = np.load(data_path, allow_pickle=True)
        assert data.shape[-1] == 3, 'The shape of the data: %r' % (data.shape)

        if encoding_path is None:
            if train is None:
                encoding = None
            else:
                encoding = {
                    'train': None,
                    'test': None,
                }
        else:
            logging.info(
                'Loading the vqvae encoding from %s...', encoding_path)
            encoding = np.load(encoding_path, allow_pickle=True)

        if train is None:
            self.data = data
            self.encoding = encoding
        else:
            data = data.item()

            if encoding is not None:
                encoding = encoding.item()

            if train is True:
                self.data = data['train']
                self.encoding = encoding['train']
            else:
                self.data = data['test']
                self.encoding = encoding['test']

        if is_val_format:
            self.data = np.reshape(self.data,
                                   [num_samples, num_steps, 3, 48, 48])
            self.data = np.transpose(self.data, [0, 1, 4, 3, 2])  # TODO

        # Preprocess the data.
        if crop is not None:
            self.data = self.data[..., :crop[0], :crop[1], :]

        if preprocess_image:
            self.data = self.data.astype(np.float32) / 255 - 0.5

        if channel_first:
            if self.data.ndim == 4:
                self.data = np.transpose(self.data, (0, 3, 1, 2))
            elif self.data.ndim == 5:
                self.data = np.transpose(self.data, (0, 1, 4, 2, 3))

        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        data_i = self.data[idx]

        if self.transform is not None:
            data_i = self.transform(data_i)

        if self.encoding is None:
            return data_i
        else:
            encoding_i = self.encoding[idx]
            return {
                's': data_i,
                'h': encoding_i,
            }


class VaeGoalDataset(VaeDataset):

    def __init__(self,
                 data_path,
                 encoding_path=None,
                 train=None,
                 vqvae=None,
                 transform=None,
                 crop=None,
                 preprocess_image=True,
                 delta_t=1,
                 channel_first=True):
        super(VaeGoalDataset, self).__init__(
            data_path=data_path,
            encoding_path=encoding_path,
            train=train,
            vqvae=vqvae,
            transform=transform,
            crop=crop,
            preprocess_image=preprocess_image,
            channel_first=channel_first)

        assert len(self.data.shape) == 5
        self.num_steps = self.data.shape[1]
        self.delta_t = delta_t

    def __getitem__(self, idx):
        t0 = int(np.random.uniform(0, self.num_steps - self.delta_t))
        t1 = t0 + self.delta_t

        data_i = np.stack(
            [self.data[idx, t0, ...], self.data[idx, t1, ...]],
            axis=0)

        if self.transform is not None:
            data_i = self.transform(data_i)

        if self.encoding is None:
            return data_i
        else:
            encoding_i = np.stack(
                [self.encoding[idx, t0, ...], self.encoding[idx, t1, ...]],
                axis=0)
            return {
                's': data_i,
                'h': encoding_i,
            }


class VaeContrastiveGoalDataset(VaeDataset):

    def __init__(self,
                 data_path,
                 encoding_path=None,
                 train=None,
                 transform=None,
                 crop=None,
                 preprocess_image=True,
                 delta_t=1,
                 channel_first=True):
        super(VaeContrastiveGoalDataset, self).__init__(
            data_path=data_path,
            encoding_path=encoding_path,
            train=train,
            transform=transform,
            crop=crop,
            preprocess_image=preprocess_image,
            channel_first=channel_first)

        assert len(self.data.shape) == 5
        self.num_steps = self.data.shape[1]
        self.delta_t = delta_t

    def __getitem__(self, idx):
        t0 = int(np.random.uniform(0, self.num_steps - self.delta_t))
        t1 = t0 + self.delta_t

        t2_candidates = np.concatenate(
            [np.arange(0, t0 - self.delta_t),
             np.arange(t1 + self.delta_t, self.num_steps)],
            axis=0)
        t2 = np.random.choice(t2_candidates)

        data_i = np.stack(
            [self.data[idx, t0, ...],
             self.data[idx, t1, ...],
             self.data[idx, t2, ...]],
            axis=0)

        if self.transform is not None:
            data_i = self.transform(data_i)

        if self.encoding is None:
            return data_i
        else:
            encoding_i = np.stack(
                [self.encoding[idx, t0, ...],
                 self.encoding[idx, t1, ...],
                 self.encoding[idx, t2, ...]],
                axis=0)
            return {
                's': data_i,
                'h': encoding_i,
            }


class VaeMultistepDataset(VaeDataset):

    def __init__(self,
                 data_path,
                 encoding_path=None,
                 train=None,
                 transform=None,
                 crop=None,
                 preprocess_image=True,
                 delta_t=1,
                 num_goals=5,
                 channel_first=True):
        super(VaeMultistepDataset, self).__init__(
            data_path=data_path,
            encoding_path=encoding_path,
            train=train,
            transform=transform,
            crop=crop,
            preprocess_image=preprocess_image,
            channel_first=channel_first)

        assert len(self.data.shape) == 5
        self.num_steps = self.data.shape[1]
        self.delta_t = delta_t
        self.num_goals = num_goals

    def __getitem__(self, idx):
        total_steps = self.delta_t * self.num_goals
        min_step = 0
        max_step = self.num_steps - total_steps
        t0 = int(np.random.uniform(min_step, max_step))

        data_i = self.data[idx, t0:t0+total_steps+1:self.delta_t]

        if self.transform is not None:
            data_i = self.transform(data_i)

        if self.encoding is None:
            return data_i
        else:
            encoding_i = self.encoding[idx, t0:t0+total_steps+1:self.delta_t]
            return {
                's': data_i,
                'h': encoding_i,
            }


########################################
# Below are deprecated.
########################################


class StepVaeDataset(VaeDataset):

    def __init__(self,
                 file_path,
                 step=0,
                 train=True,
                 transform=None,
                 crop=None,
                 preprocess_image=True):
        data = np.load(file_path, allow_pickle=True)
        data = data.item()

        if train is None:
            self.data = data
        elif train is True:
            self.data = data['train']
        else:
            self.data = data['test'][:, step]

        if crop is not None:
            self.data = self.data[..., :crop[0], :crop[1], :]

        if preprocess_image:
            self.data = self.data.astype(np.float32) / 255 - 0.5

        self.transform = transform


class ValDataset(VaeDataset):

    def __init__(self,
                 file_path,
                 train=True,
                 transform=None,
                 crop=None,
                 preprocess_image=True,
                 num_goal_steps=15,
                 ):
        del train
        del crop
        data = np.load(file_path, allow_pickle=True)
        data = data.item()

        self.data = data['observations']

        num_samples = self.data.shape[0]
        num_steps = self.data.shape[1]
        self.data = np.reshape(self.data, [num_samples, num_steps, 3, 48, 48])
        self.data = np.transpose(self.data, [0, 1, 4, 3, 2])  # TODO
        self.num_steps = num_steps
        self.num_goal_steps = num_goal_steps

        if preprocess_image:
            self.data = self.data.astype(np.float32) / 255 - 0.5

        self.transform = transform


class ValTransitionDataset(ValDataset):

    def __getitem__(self, idx):
        t0 = int(np.random.uniform(0, self.num_steps - self.num_goal_steps))
        t1 = t0 + self.num_goal_steps

        data_i = np.stack(
            [self.data[idx, t0, ...], self.data[idx, t1, ...]],
            axis=0)

        if self.transform is not None:
            data_i = self.transform(data_i)

        return data_i
