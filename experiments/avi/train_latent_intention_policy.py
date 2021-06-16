import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch.distributions as td

import numpy as np
import os
import argparse
import time

from rlkit.core.timer import timer
import rlkit.torch.pytorch_util as ptu
from rlkit.torch.networks.cnn import CNN, ConcatCNN
from rlkit.launchers.config import LOCAL_LOG_DIR
from rlkit.core import logger
from rlkit.launchers.launcher_util import setup_logger

BUFFER = '/media/avi/data/Work/github/avisingh599/minibullet/data/june15_test_Widow250PickPlaceMedium-v0_100_noise_0.1_2021-06-15T14-36-15/june15_test_Widow250PickPlaceMedium-v0_100_noise_0.1_2021-06-15T14-36-15_100.npy'


class DataLoader:

    def __init__(self, data, batch_size=32):
        self.dataset_size = len(data)
        self.batch_size = batch_size
        self.trajectory_length = len(data[0]['actions'])
        self.action_sequences = []

        for i in range(self.dataset_size):
            self.action_sequences.append(data[i]['actions'])

        self.action_sequences = np.asarray(self.action_sequences)
        self.data = data

    def get_batch(self):
        trajectory_indices = np.random.randint(self.dataset_size, size=self.batch_size)
        timestep_indices = np.random.randint(self.trajectory_length, size=self.batch_size)

        input_image_observations = []
        input_state_observations = []
        target_actions = []

        for i in range(self.batch_size):
            j = trajectory_indices[i]
            k = timestep_indices[i]

            input_image_observations.append(data[j]['observations'][k]['image'])
            input_state_observations.append(data[j]['observations'][k]['state'])
            target_actions.append(data[j]['actions'][k])

        input_image_observations = np.asarray(input_image_observations)
        input_state_observations = np.asarray(input_state_observations)
        target_actions = np.asarray(target_actions)

        batch = dict(
            input_action_sequence=ptu.from_numpy(self.action_sequences[trajectory_indices]),
            input_image_observations=ptu.from_numpy(input_image_observations),
            input_state_observations=ptu.from_numpy(input_state_observations),
            target_actions=ptu.from_numpy(target_actions)
        )

        return batch


class TrajectoryConditionedPolicy(nn.Module):

    def __init__(self, action_dim, latent_dim, rnn_hidden_size, cnn_params,
                 trajectory_length, batch_size=32, ignore_z=False, use_fc=False):

        super(TrajectoryConditionedPolicy, self).__init__()

        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.rnn_hidden_size = rnn_hidden_size
        self.trajectory_length = trajectory_length
        self.batch_size = batch_size
        self.ignore_z = ignore_z
        self.use_fc = use_fc

        # sequence encoder
        if not self.use_fc:
            self.gru = nn.GRU(action_dim, rnn_hidden_size, batch_first=True)
        else:
            fc_input_size = self.trajectory_length*self.action_dim
            self.fc1 = nn.Linear(fc_input_size, rnn_hidden_size)
            self.fc2 = nn.Linear(rnn_hidden_size, rnn_hidden_size)

        # sequence to mu, var
        self.fc_mu = nn.Linear(rnn_hidden_size, latent_dim)
        self.fc_var = nn.Linear(rnn_hidden_size, latent_dim)

        # (obs, z) to action
        self.cnn = ConcatCNN(**cnn_params)

    def forward(self, input):
        hidden = self.init_hidden(self.batch_size)
        if not self.use_fc:
            rnn_output, rnn_hidden = self.gru(input['input_action_sequence'], hidden)
            sequence_encoding = rnn_hidden[0]

        else:
            x = input['input_action_sequence'].view(self.batch_size, -1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            sequence_encoding = x

        # import IPython; IPython.embed()
        mu = self.fc_mu(sequence_encoding)
        log_var = self.fc_var(sequence_encoding)

        z, q_z = self.reparameterize(mu, log_var)

        if self.ignore_z:
            output = self.cnn(input['input_image_observations'],
                              input['input_state_observations'],)
        else:
            output = self.cnn(input['input_image_observations'],
                              input['input_state_observations'],
                              z)
        return output, q_z

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.rnn_hidden_size, device=ptu.device)

    def get_action(self, obs, z):
        obs_image = ptu.from_numpy(np.expand_dims(obs['image'], axis=0))
        obs_state = ptu.from_numpy(np.expand_dims(obs['state'], axis=0))
        z = ptu.from_numpy(np.expand_dims(z, axis=0))

        return ptu.get_numpy(self.cnn(obs_image, obs_state, z))[0]

    def reparameterize(self, mu, logvar):
        """
        From https://github.com/AntixK/PyTorch-VAE/blob/master/models/beta_vae.py
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        q_z = td.normal.Normal(mu, std)     # create a torch distribution
        eps = torch.randn_like(std)
        z = eps * std + mu
        return z, q_z


def enable_gpus(gpu_str):
    if gpu_str != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer", type=str, default=BUFFER)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--beta-target", type=float, default=0.01)
    parser.add_argument("--ignore-z", default=False, action='store_true')
    parser.add_argument("--use-fc", default=False, action='store_true')
    parser.add_argument("--gpu", default='0', type=str)
    args = parser.parse_args()

    enable_gpus(args.gpu)
    ptu.set_gpu_mode(True)

    state_observation_dim = 10
    action_dim = 8
    latent_dim = 5
    batch_size = args.batch_size

    total_steps = int(1e5)
    log_freq = 1000
    half_beta_target_steps = min(total_steps // 2, 25000)
    beta_target = args.beta_target

    variant = dict(
        buffer=args.buffer,
        use_fc=args.use_fc,
        action_dim=action_dim,
        latent_dim=latent_dim,
        rnn_hidden_size=512,
        ignore_z=args.ignore_z,
        batch_size=batch_size,
        beta_target=beta_target,
        cnn_params=dict(
            input_width=48,
            input_height=48,
            input_channels=3,

            kernel_sizes=[3, 3, 3],
            n_channels=[16, 16, 16],
            strides=[1, 1, 1],
            hidden_sizes=[1024, 512, 256],
            paddings=[1, 1, 1],
            pool_type='max2d',
            pool_sizes=[2, 2, 1],  # the one at the end means no pool
            pool_strides=[2, 2, 1],
            pool_paddings=[0, 0, 0],

            image_augmentation=True,
            image_augmentation_padding=4,

            output_size=action_dim,
            added_fc_input_size=state_observation_dim+latent_dim,
        )
    )

    if args.ignore_z:
        variant['cnn_params']['added_fc_input_size'] = state_observation_dim

    # LOG_FOLDER = '{}-latent_intention_model'.format(time.strftime("%y-%m-%d"))
    # save_dir = os.path.join(LOCAL_LOG_DIR, LOG_FOLDER)
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)

    exp_prefix = '{}-latent_intention_model'.format(time.strftime("%y-%m-%d"))
    setup_logger(logger, exp_prefix, LOCAL_LOG_DIR, variant=variant,
                 snapshot_mode='gap_and_last', snapshot_gap=10, )

    with open(args.buffer, 'rb') as fl:
        data = np.load(fl, allow_pickle=True)
    train_size = int(0.8*len(data))
    train_dataloader = DataLoader(data[:train_size], batch_size=batch_size)
    val_dataloader = DataLoader(data[train_size:], batch_size=batch_size)

    seq_cond_policy = TrajectoryConditionedPolicy(
        action_dim=action_dim,
        latent_dim=latent_dim,
        rnn_hidden_size=variant['rnn_hidden_size'],
        trajectory_length=train_dataloader.trajectory_length,
        cnn_params=variant['cnn_params'],
        batch_size=variant['batch_size'],
        ignore_z=variant['ignore_z']
    )

    seq_cond_policy.to(ptu.device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(seq_cond_policy.parameters(), lr=3e-4)

    p_z = td.normal.Normal(ptu.from_numpy(np.zeros((latent_dim,))),
                           ptu.from_numpy(np.ones((latent_dim,))))
    running_loss_mse = 0.
    running_loss_kl = 0.

    start_time = time.time()

    for i in range(total_steps):

        optimizer.zero_grad()
        batch = train_dataloader.get_batch()
        predicted_actions, q_z = seq_cond_policy(batch)
        mse_loss = criterion(predicted_actions, batch['target_actions'])
        KLD = td.kl_divergence(q_z, p_z).sum()

        def kl_anneal_function(step, x0, k=0.0025):
            return float(1 / (1 + np.exp(-k * (step - x0))))

        beta = beta_target*kl_anneal_function(i, half_beta_target_steps)

        loss = mse_loss + beta*KLD
        loss.backward()
        optimizer.step()
        running_loss_mse += mse_loss.item()
        running_loss_kl += KLD.item()

        if i % log_freq == 0:

            logger.record_tabular('steps', i)

            if i > 0:
                running_loss_kl /= log_freq
                running_loss_mse /= log_freq
                logger.record_tabular('time/epoch_time', time.time() - start_time)
                start_time = time.time()
            else:
                logger.record_tabular('time/epoch_time', 0.0)

            logger.record_tabular('train/mse_loss', running_loss_mse)
            logger.record_tabular('train/kl_loss', running_loss_kl)

            num_val_batches = 10
            mse_loss_val = 0.
            KLD_val = 0.
            for i in range(num_val_batches):
                batch = val_dataloader.get_batch()
                predicted_actions, q_z = seq_cond_policy(batch)
                mse_loss_val += criterion(predicted_actions, batch['target_actions']).item()
                KLD_val += td.kl_divergence(q_z, p_z).sum().item()

            logger.record_tabular('val/mse_loss', mse_loss_val/num_val_batches)
            logger.record_tabular('val/kl_loss', KLD_val/num_val_batches)

            # save_path = os.path.join(save_dir, 'step_{}.pkl'.format(i))
            # print('checkpoint', save_path)
            # torch.save(seq_cond_policy.state_dict(), save_path)
            params = seq_cond_policy.state_dict()
            logger.save_itr_params(i, params)

            running_loss_mse = 0.
            running_loss_kl = 0.
            logger.dump_tabular(with_prefix=True, with_timestamp=False)
