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


class TransformerTrajectoryEncoderTrainer:

    def __init__(self, net, optimizer, criterion, print_freq, save_freq,
                 beta_target, half_beta_target_steps, anneal, encoder_keys=['observations'],
                 ):
        self.net = net
        self.optimizer = optimizer
        self.criterion = criterion
        self.print_freq = print_freq
        self.save_freq = save_freq
        self.beta_target = beta_target
        self.half_beta_target_steps = half_beta_target_steps
        self.anneal = anneal
        self.encoder_keys = encoder_keys

    def train(self, replay_buffer, replay_buffer_val,
              total_steps, batch_size, tasks, logger):

        val_batch_size = batch_size * 2
        start_time = time.time()
        num_tasks = len(tasks)


        for i in range(total_steps):
            self.optimizer.zero_grad()
            encoder_batch = replay_buffer.sample_batch_of_trajectories([0], batch_size)
            encoder_batch_traj = [ptu.from_numpy(encoder_batch[k]) for k in self.encoder_keys]
           
            _, _, path_len, obs_dim = encoder_batch['observations'].shape 
            action_dim = encoder_batch['actions'].shape[-1]
            decoder_idxs = torch.randint(path_len, size=(1, batch_size, 1, 1)).cuda()
            decoder_obs = torch.gather(encoder_batch_traj[0], dim=2, 
                index=decoder_idxs.repeat(1, 1, 1, obs_dim)).squeeze(dim=2)

            out = self.net.forward(
                encoder_batch_traj, decoder_obs)
            
            action_predictions, mu, logvar = out
            for e in encoder_batch_traj:
                e.detach().cpu()
            decoder_obs.detach().cpu()
            torch.cuda.empty_cache()

            if self.anneal == 'sigmoid':
                beta = self.beta_target*kl_anneal_sigmoid_function(i, self.half_beta_target_steps)
            elif self.anneal == 'linear':
                beta = self.beta_target*kl_anneal_linear_function(i, self.half_beta_target_steps)
            elif self.anneal == 'none':
                beta = self.beta_target
            else:
                raise NotImplementedError
            gamma = 1 - kl_anneal_linear_function(i, self.half_beta_target_steps)
            
            encoder_actions = ptu.from_numpy(encoder_batch['actions'])
            decoder_actions = torch.gather(encoder_actions, dim=2, 
                index=decoder_idxs.repeat(1, 1, 1, action_dim)).squeeze(dim=2)
      
            t, b, action_dim = decoder_actions.shape
            decoder_actions = decoder_actions.view(t*b, action_dim)

            mse_loss = self.criterion(action_predictions, decoder_actions) 
            var = torch.exp(logvar)
            KLD = torch.sum(-logvar + (mu ** 2)*0.5 + var, 1) - self.net.latent_dim
            KLD = KLD.mean()
            # KLD = td.kl_divergence(q_z, p_z).sum()

            loss = mse_loss + beta * KLD

            loss.backward()
            self.optimizer.step()

            if i % self.print_freq == 0:

                logger.record_tabular('steps', i)
                logger.record_tabular('train/beta', beta)

                if i > 0:
                    logger.record_tabular('time/epoch_time', time.time() - start_time)
                    start_time = time.time()
                else:
                    logger.record_tabular('time/epoch_time', 0.0)

                logger.record_tabular('train/mse_loss', mse_loss.item())
                logger.record_tabular('train/loss', loss.item())
                logger.record_tabular('train/KLD', KLD.item())

                #Get validation statistics
                encoder_batch_val = replay_buffer_val.sample_batch_of_trajectories([0], val_batch_size)
                encoder_batch_val_traj = [ptu.from_numpy(encoder_batch_val[k]) for k in self.encoder_keys]
                decoder_idxs = torch.randint(path_len, size=(1, val_batch_size, 1, 1)).cuda()
                decoder_batch_val_obs = torch.gather(encoder_batch_val_traj[0], dim=2, 
                    index=decoder_idxs.repeat(1, 1, 1, obs_dim)).squeeze(dim=2)
                
                with torch.no_grad():
                    out = self.net.forward(
                        encoder_batch_val_traj,
                        decoder_batch_val_obs)
                    action_predictions, mu, logvar = out
                    for e in encoder_batch_val_traj:
                        e.detach().cpu()
                    decoder_batch_val_obs.detach().cpu()
                torch.cuda.empty_cache()
               
                encoder_val_actions = ptu.from_numpy(encoder_batch_val['actions']) 
                decoder_actions = torch.gather(encoder_val_actions, dim=2,  
                    index=decoder_idxs.repeat(1, 1, 1, action_dim)).squeeze(dim=2)
                
                t, b, action_dim = decoder_actions.shape
                decoder_actions = decoder_actions.view(t*b,action_dim)

                mse_loss = self.criterion(action_predictions, decoder_actions)
                var = torch.exp(logvar)
                KLD = torch.sum(-logvar + (mu ** 2)*0.5 + var, 1) - self.net.latent_dim
                KLD = KLD.mean()

                loss = mse_loss + beta * KLD

                mu_max = mu.max()
                mu_min = mu.min()
                logger.record_tabular('val/mu/min', mu_min.item())
                logger.record_tabular('val/mu/max', mu_max.item())

                logger.record_tabular('val/mse_loss', mse_loss.item())
                logger.record_tabular('val/KLD', KLD.item())
                logger.record_tabular('val/loss', loss.item())

                # print('accuracy', accuracy_score(gt_rewards, reward_predictions))
                # print('recall', recall_score(gt_rewards, reward_predictions))
                # print('precision', precision_score(gt_rewards, reward_predictions))
                params = self.net.state_dict()
                if i % (self.print_freq*self.save_freq) == 0:
                    plt.clf()
                    mu_np = ptu.get_numpy(mu)
                    mu_np = np.reshape(mu_np, (val_batch_size, self.net.latent_dim))
                    plt.scatter(mu_np[:, 0], mu_np[:, 1], label=0, s=3)
                    save_path = osp.join(logger._snapshot_dir, 'plot_{}.pdf'.format(i//self.print_freq))
                    plt.legend()
                    plt.savefig(save_path)

                logger.save_itr_params(i // self.print_freq, params)
                logger.dump_tabular(with_prefix=True, with_timestamp=False)
