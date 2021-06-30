import os
import numpy as np
import argparse
import time

from rlkit.data_management.multitask_replay_buffer import ObsDictMultiTaskReplayBuffer
from rlkit.misc.roboverse_utils import add_reward_filtered_data_to_buffers_multitask, get_buffer_size_multitask, process_keys
import roboverse
import rlkit.torch.pytorch_util as ptu

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, precision_score, recall_score
from rlkit.launchers.config import LOCAL_LOG_DIR
from rlkit.core import logger
from rlkit.launchers.launcher_util import setup_logger

def enable_gpus(gpu_str):
    if gpu_str != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
    return

BUFFER = ('/nfs/kun1/users/avi/scripted_sim_datasets/june2_Widow250PickPlace'
          'MetaTrainMultiObjectMultiContainer-v0_16K_save_all_noise_0.1_2021-'
          '06-02T16-20-10/june2_Widow250PickPlaceMetaTrainMultiObjectMulti'
          'Container-v0_16K_save_all_noise_0.1_2021-06-02T16-20-10_16000.npy')
VALIDATION_BUFFER = ('/nfs/kun1/users/jonathan/minibullet_data/'
                     'jun_14_validation_Widow250PickPlaceMetaTrainMultiObjectMultiContainer-v0_1K_save_all_noise_0.1_2021-06-14T11-27-40/'
                     'jun_14_validation_Widow250PickPlaceMetaTrainMultiObjectMultiContainer-v0_1K_save_all_noise_0.1_2021-06-14T11-27-40_1024.npy')
ENV = 'Widow250PickPlaceMetaTestMultiObjectMultiContainer-v0'


class EncoderNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*9*9, 512)

        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, encoder_input):
        t, b, obs_dim = encoder_input.shape
        x = encoder_input.view(t*b, obs_dim)
        x = x.view(t*b, 3, 48, 48)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DecoderNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_output_dim = 10

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*9*9 + self.encoder_output_dim, 512)

        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2)

    def forward(self, task_embedding, decoder_input):

        t, b, obs_dim = decoder_input.shape
        x = decoder_input.view(t*b, obs_dim)
        x = x.view(t*b, 3, 48, 48)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = torch.cat((x, task_embedding), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class EncoderDecoderNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder_net = EncoderNet()
        self.decoder_net = DecoderNet()

    def forward(self, encoder_input, decoder_input):
        task_embedding = self.encoder_net(encoder_input)
        predicted_reward = self.decoder_net(task_embedding, decoder_input)
        return predicted_reward


def main(args):
    variant = dict(
        buffer=args.buffer,
        val_buffer=args.val_buffer,
        batch_size=128,
        num_tasks=8,
    )

    enable_gpus(args.gpu)
    ptu.set_gpu_mode(True)

    train_task_indices = list(range(32))
    with open(variant['buffer'], 'rb') as fl:
        data = np.load(fl, allow_pickle=True)

    observation_keys = ['image']

    num_transitions = get_buffer_size_multitask(data)
    max_replay_buffer_size = num_transitions + 10
    expl_env = roboverse.make(ENV, transpose_image=True)

    replay_buffer_positive = ObsDictMultiTaskReplayBuffer(
        int(max_replay_buffer_size/2),
        expl_env,
        train_task_indices,
        use_next_obs_in_context=False,
        sparse_rewards=False,
        observation_keys=observation_keys
    )

    replay_buffer_full = ObsDictMultiTaskReplayBuffer(
        max_replay_buffer_size,
        expl_env,
        train_task_indices,
        use_next_obs_in_context=False,
        sparse_rewards=False,
        observation_keys=observation_keys
    )

    add_reward_filtered_data_to_buffers_multitask(data, observation_keys,
                                                  (replay_buffer_positive, lambda r: r > 0),
                                                  (replay_buffer_full, lambda r: True))
    # train_task_indices = list(range(32))
    with open(variant['val_buffer'], 'rb') as fl:
        data = np.load(fl, allow_pickle=True)

    observation_keys = ['image']
    num_transitions = get_buffer_size_multitask(data)
    max_replay_buffer_size = num_transitions + 10
    # expl_env = roboverse.make(ENV, transpose_image=True)

    replay_buffer_positive_val = ObsDictMultiTaskReplayBuffer(
        int(max_replay_buffer_size/2),
        expl_env,
        train_task_indices,
        use_next_obs_in_context=False,
        sparse_rewards=False,
        observation_keys=observation_keys
    )
    replay_buffer_full_val = ObsDictMultiTaskReplayBuffer(
        max_replay_buffer_size,
        expl_env,
        train_task_indices,
        use_next_obs_in_context=False,
        sparse_rewards=False,
        observation_keys=observation_keys
    )
    add_reward_filtered_data_to_buffers_multitask(data, observation_keys,
                                                  (replay_buffer_positive_val, lambda r: r > 0),
                                                  (replay_buffer_full_val, lambda r: True))

    net = EncoderDecoderNet()
    net.to(ptu.device)
    exp_prefix = '{}-task-encoder-decoder'.format(time.strftime("%y-%m-%d"))
    setup_logger(logger, exp_prefix, LOCAL_LOG_DIR, variant=variant,
                 snapshot_mode='gap_and_last', snapshot_gap=10, )

    running_loss = 0.
    # running_loss_kl = 0.
    batch_size = variant['batch_size']
    val_batch_size = batch_size*4

    optimizer = optim.Adam(net.parameters(), lr=3e-4)
    # criterion = nn.MSELoss()
    print_freq = 1000
    criterion = nn.CrossEntropyLoss()
    tasks_to_sample = list(range(variant['num_tasks']))
    start_time = time.time()

    for i in range(50000):
        optimizer.zero_grad()
        encoder_batch = replay_buffer_positive.sample_batch(tasks_to_sample, batch_size)
        # positives
        decoder_batch_1 = replay_buffer_positive.sample_batch(tasks_to_sample, batch_size // 2)
        # random transitions
        decoder_batch_2 = replay_buffer_full.sample_batch(tasks_to_sample, batch_size // 4)
        # hard negatives
        decoder_batch_3 = replay_buffer_positive.sample_batch(tasks_to_sample, batch_size // 4)
        np.random.shuffle(decoder_batch_3['observations'])
        # decoder_batch = replay_buffer_full.sample_batch(tasks_to_sample, batch_size)

        decoder_obs = np.concatenate((decoder_batch_1['observations'],
                                      decoder_batch_2['observations'],
                                      decoder_batch_3['observations']
                                      ),
                                     axis=1)

        reward_predictions = net.forward(ptu.from_numpy(encoder_batch['observations']),
                                         ptu.from_numpy(decoder_obs)
                                         )
        # gt_rewards = torch.from_numpy(decoder_batch['rewards'].astype(np.int64)).cuda()
        decoder_rewards = np.concatenate(
            (decoder_batch_1['rewards'],
             decoder_batch_2['rewards'],
             np.zeros_like(decoder_batch_3['rewards'])),
            axis=1)
        gt_rewards = torch.from_numpy(decoder_rewards.astype(np.int64)).cuda()

        t, b, rew_dim = gt_rewards.shape
        assert rew_dim == 1
        gt_rewards = gt_rewards.view(t*b,)

        loss = criterion(reward_predictions, gt_rewards)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

        if i%print_freq == 0:

            logger.record_tabular('steps', i)

            if i > 0:
                running_loss /= print_freq
                logger.record_tabular('time/epoch_time', time.time() - start_time)
                start_time = time.time()
            else:
                logger.record_tabular('time/epoch_time', 0.0)

            # print('steps: {} train loss: {}'.format(i, running_loss))
            logger.record_tabular('train/entropy_loss', running_loss)

            running_loss = 0.0

            encoder_batch_val = replay_buffer_positive_val.sample_batch(tasks_to_sample, val_batch_size)
            decoder_batch_val = replay_buffer_full_val.sample_batch(tasks_to_sample, val_batch_size)
            reward_predictions = net.forward(ptu.from_numpy(encoder_batch_val['observations']),
                                             ptu.from_numpy(decoder_batch_val['observations'])
                                             )
            # gt_rewards = ptu.from_numpy(decoder_batch_val['rewards'])
            # t, b, rew_dim = gt_rewards.shape
            # gt_rewards = gt_rewards.view(t*b, rew_dim)
            gt_rewards = torch.from_numpy(decoder_batch_val['rewards'].astype(np.int64)).cuda()
            # gt_rewards = torch.squeeze(gt_rewards, dim=1)
            t, b, rew_dim = gt_rewards.shape
            assert rew_dim == 1
            gt_rewards = gt_rewards.view(t*b,)
            loss = criterion(reward_predictions, gt_rewards)
            # print('steps: {} val loss: {}'.format(i, loss.item()))
            logger.record_tabular('val/entropy_loss', loss.item())

            gt_rewards = ptu.get_numpy(gt_rewards)
            reward_predictions = ptu.get_numpy(reward_predictions)
            reward_predictions = np.argmax(reward_predictions, axis=1)
            logger.record_tabular('val/accuracy', accuracy_score(gt_rewards, reward_predictions))
            logger.record_tabular('val/precision', precision_score(gt_rewards, reward_predictions))
            logger.record_tabular('val/recall', recall_score(gt_rewards, reward_predictions))

            # print('accuracy', accuracy_score(gt_rewards, reward_predictions))
            # print('recall', recall_score(gt_rewards, reward_predictions))
            # print('precision', precision_score(gt_rewards, reward_predictions))
            logger.dump_tabular(with_prefix=True, with_timestamp=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer", type=str, default=BUFFER)
    parser.add_argument("--val-buffer", type=str, default=VALIDATION_BUFFER)
    parser.add_argument("--gpu", default='0', type=str)
    args = parser.parse_args()
    main(args)