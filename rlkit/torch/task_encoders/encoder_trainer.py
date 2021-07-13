import matplotlib
matplotlib.use('Agg')
import time
import os.path as osp

import torch

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt

import rlkit.torch.pytorch_util as ptu


def kl_anneal_sigmoid_function(step, x0, k=0.0025):
    return float(1 / (1 + np.exp(-k * (step - x0))))


def kl_anneal_linear_function(step, x0):
    return min(1.0, step/x0)


class TaskEncoderTrainer:

    def __init__(self, net, optimizer, criterion, print_freq, save_freq,
                 beta_target, half_beta_target_steps, anneal):
        self.net = net
        self.optimizer = optimizer
        self.criterion = criterion
        self.print_freq = print_freq
        self.save_freq = save_freq
        self.beta_target = beta_target
        self.half_beta_target_steps = half_beta_target_steps
        self.anneal = anneal

    def train(self, replay_buffer_full, replay_buffer_positive,
              replay_buffer_full_val, replay_buffer_positive_val, total_steps,
              batch_size, tasks_to_sample, logger):

        val_batch_size = batch_size*4
        running_loss_entropy = 0.
        running_loss_kl = 0.
        start_time = time.time()
        num_tasks = len(tasks_to_sample)

        for i in range(total_steps):
            self.optimizer.zero_grad()
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

            reward_predictions, mu, logvar = self.net.forward(
                ptu.from_numpy(encoder_batch['observations']), ptu.from_numpy(decoder_obs))

            if self.anneal == 'sigmoid':
                beta = self.beta_target*kl_anneal_sigmoid_function(i, self.half_beta_target_steps)
            elif self.anneal == 'linear':
                beta = self.beta_target*kl_anneal_linear_function(i, self.half_beta_target_steps)
            elif self.anneal == 'none':
                beta = self.beta_target
            else:
                raise NotImplementedError

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

            entropy_loss = self.criterion(reward_predictions, gt_rewards)
            var = torch.exp(logvar)
            KLD = torch.sum(-logvar + (mu ** 2)*0.5 + var, 1) - self.net.latent_dim
            KLD = KLD.mean()
            # KLD = td.kl_divergence(q_z, p_z).sum()

            loss = entropy_loss + beta*KLD

            running_loss_kl += KLD.item()
            running_loss_entropy += entropy_loss.item()
            loss.backward()
            self.optimizer.step()

            if i % self.print_freq == 0:

                logger.record_tabular('steps', i)
                logger.record_tabular('train/beta', beta)

                if i > 0:
                    running_loss_entropy /= self.print_freq
                    running_loss_kl /= self.print_freq
                    logger.record_tabular('time/epoch_time', time.time() - start_time)
                    start_time = time.time()
                else:
                    logger.record_tabular('time/epoch_time', 0.0)

                # print('steps: {} train loss: {}'.format(i, running_loss))
                logger.record_tabular('train/entropy_loss', running_loss_entropy)
                logger.record_tabular('train/KLD', running_loss_kl)
                running_loss_entropy = 0.
                running_loss_kl = 0.

                encoder_batch_val = replay_buffer_positive_val.sample_batch(tasks_to_sample, val_batch_size)
                decoder_batch_val = replay_buffer_full_val.sample_batch(tasks_to_sample, val_batch_size)
                reward_predictions, mu, logvar = self.net.forward(
                    ptu.from_numpy(encoder_batch_val['observations']),
                    ptu.from_numpy(decoder_batch_val['observations']))

                # gt_rewards = ptu.from_numpy(decoder_batch_val['rewards'])
                # t, b, rew_dim = gt_rewards.shape
                # gt_rewards = gt_rewards.view(t*b, rew_dim)
                gt_rewards = torch.from_numpy(decoder_batch_val['rewards'].astype(np.int64)).cuda()
                # gt_rewards = torch.squeeze(gt_rewards, dim=1)
                t, b, rew_dim = gt_rewards.shape
                assert rew_dim == 1
                gt_rewards = gt_rewards.view(t*b,)
                entropy_loss = self.criterion(reward_predictions, gt_rewards)
                # KLD = td.kl_divergence(q_z, p_z).sum()
                var = torch.exp(logvar)
                KLD = torch.sum(-logvar + (mu ** 2)*0.5 + var, 1) - self.net.latent_dim
                KLD = KLD.mean()

                mu_max = mu.max()
                mu_min = mu.min()
                logger.record_tabular('val/mu/min', mu_min.item())
                logger.record_tabular('val/mu/max', mu_max.item())

                logger.record_tabular('val/entropy_loss', entropy_loss.item())
                logger.record_tabular('val/KLD', KLD.item())

                gt_rewards = ptu.get_numpy(gt_rewards)
                reward_predictions = ptu.get_numpy(reward_predictions)
                reward_predictions = np.argmax(reward_predictions, axis=1)
                logger.record_tabular('val/accuracy', accuracy_score(gt_rewards, reward_predictions))
                logger.record_tabular('val/precision', precision_score(gt_rewards, reward_predictions))
                logger.record_tabular('val/recall', recall_score(gt_rewards, reward_predictions))

                # print('accuracy', accuracy_score(gt_rewards, reward_predictions))
                # print('recall', recall_score(gt_rewards, reward_predictions))
                # print('precision', precision_score(gt_rewards, reward_predictions))
                params = self.net.state_dict()
                if i % (self.print_freq*self.save_freq) == 0:
                    plt.clf()
                    mu_np = ptu.get_numpy(mu)
                    mu_np = np.reshape(mu_np, (num_tasks, val_batch_size, self.net.latent_dim))
                    for j in range(num_tasks):
                        plt.scatter(mu_np[j, :, 0], mu_np[j, :, 1], label=j)
                    save_path = osp.join(logger._snapshot_dir, 'plot_{}.pdf'.format(i//self.print_freq))
                    plt.savefig(save_path)

                logger.save_itr_params(i // self.print_freq, params)
                logger.dump_tabular(with_prefix=True, with_timestamp=False)
