from collections import OrderedDict, namedtuple
from itertools import chain
from numbers import Number
from typing import Tuple

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn
from torch.distributions import kl_divergence

import rlkit.torch.pytorch_util as ptu
from rlkit.core.logging import add_prefix
from rlkit.core.loss import LossStatistics
from rlkit.core.timer import timer
from rlkit.misc.eval_util import create_stats_ordered_dict
from rlkit.torch.pearl.pearl_sac import PEARLSoftActorCriticTrainer
from rlkit.torch.torch_rl_algorithm import TorchTrainer


class PearlCqlTrainer(TorchTrainer):
    def __init__(
            self,
            latent_dim,
            agent,
            qf1,
            qf2,
            target_qf1,
            target_qf2,
            context_encoder,
            reward_predictor,
            context_decoder,

            reward_scale=1.,
            discount=0.99,
            policy_lr=1e-3,
            qf_lr=1e-3,
            context_lr=1e-3,
            kl_lambda=1.,
            policy_mean_reg_weight=1e-3,
            policy_std_reg_weight=1e-3,
            policy_pre_activation_weight=0.,
            optimizer_class=optim.Adam,
            recurrent=False,
            use_information_bottleneck=True,
            use_next_obs_in_context=False,
            sparse_rewards=False,
            backprop_q_loss_into_encoder=False,
            train_context_decoder=False,

            train_reward_pred_in_unsupervised_phase=False,
            use_encoder_snapshot_for_reward_pred_in_unsupervised_phase=False,

            soft_target_tau=1e-2,
            target_update_period=1,
            plotter=None,
            render_eval_paths=False,

            # CQL
            num_qs=2,
            min_q_version=3,
            temp=1.0,
            min_q_weight=1.0,
            max_q_backup=False,
            deterministic_backup=True,
            num_random=10,
            with_lagrange=True,
            lagrange_thresh=0.0,
            use_automatic_entropy_tuning=True,
            target_entropy=None,
            action_space=None,
            policy_eval_start=0,
    ):
        super().__init__()

        self.train_encoder_decoder = True
        self.train_context_decoder = train_context_decoder
        self.reward_scale = reward_scale
        self.discount = discount
        self.soft_target_tau = soft_target_tau
        assert target_update_period == 1
        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight
        self.policy_pre_activation_weight = policy_pre_activation_weight
        self.plotter = plotter
        self.render_eval_paths = render_eval_paths
        self.backprop_q_loss_into_encoder = backprop_q_loss_into_encoder

        self.train_reward_pred_in_unsupervised_phase = train_reward_pred_in_unsupervised_phase
        self.use_encoder_snapshot_for_reward_pred_in_unsupervised_phase = (
            use_encoder_snapshot_for_reward_pred_in_unsupervised_phase
        )

        self.recurrent = recurrent
        self.latent_dim = latent_dim
        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()
        self.vib_criterion = nn.MSELoss()
        self.l2_reg_criterion = nn.MSELoss()
        self.reward_pred_criterion = nn.MSELoss()
        self.kl_lambda = kl_lambda

        self.use_information_bottleneck = use_information_bottleneck
        self.sparse_rewards = sparse_rewards
        self.use_next_obs_in_context = use_next_obs_in_context

        self.agent = agent
        self.policy = agent.policy
        self.qf1, self.qf2 = qf1, qf2
        self.target_qf1, self.target_qf2 = target_qf1, target_qf2
        self.context_encoder = context_encoder
        self.context_decoder = context_decoder
        self.reward_predictor = reward_predictor

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        if train_context_decoder:
            self.context_optimizer = optimizer_class(
                chain(
                    self.context_encoder.parameters(),
                    self.context_decoder.parameters(),
                ),
                lr=context_lr,
            )
        else:
            self.context_optimizer = optimizer_class(
                self.context_encoder.parameters(),
                lr=context_lr,
            )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )
        self.reward_predictor_optimizer = optimizer_class(
            self.reward_predictor.parameters(),
            lr=context_lr,
        )

        self.eval_statistics = None
        self._need_to_update_eval_statistics = True

        self.with_lagrange = with_lagrange
        if self.with_lagrange:
            self.target_action_gap = lagrange_thresh
            self.log_alpha_prime = ptu.zeros(1, requires_grad=True)
            self.alpha_prime_optimizer = optimizer_class(
                [self.log_alpha_prime],
                lr=qf_lr,
            )
        self.max_q_backup = max_q_backup
        self.deterministic_backup = deterministic_backup
        self.num_random = num_random
        self.num_qs = num_qs
        self.temp = temp
        self.min_q_version = min_q_version
        self.min_q_weight = min_q_weight
        self.policy_eval_start = policy_eval_start

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy is None:
                # Use heuristic value from SAC paper
                self.target_entropy = -np.prod(
                    action_space.shape).item()
            else:
                self.target_entropy = target_entropy
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )

    def train_from_torch(self, batch):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        context = batch['context']

        """
        Policy and Alpha Loss
        """
        action_distrib, p_z, task_z_with_grad = self.agent(
            obs, context, return_latent_posterior_and_task_z=True,
        )
        task_z_detached = task_z_with_grad.detach()
        new_obs_actions, log_pi = (
            action_distrib.rsample_and_logprob())
        log_pi = log_pi.unsqueeze(1)
        next_action_distrib = self.agent(next_obs, context)
        new_next_actions, new_log_pi = (
            next_action_distrib.rsample_and_logprob())
        new_log_pi = new_log_pi.unsqueeze(1)

        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)
        unscaled_rewards_flat = rewards.view(t * b, 1)
        rewards_flat = unscaled_rewards_flat * self.reward_scale
        terminals = terminals.view(t * b, 1)

        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (
                    log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = 1

        q_new_actions = self.qf1(obs, new_obs_actions, task_z_detached)
        if self._num_train_steps < self.policy_eval_start:
            """
            For the initial few epochs, try doing behaivoral cloning, if needed
            conventionally, there's not much difference in performance with having 20k 
            gradient steps here, or not having it
            """
            policy_log_prob = action_distrib.log_prob(obs, actions)
            policy_loss = (alpha * log_pi - policy_log_prob).mean()
        else:
            policy_loss = (alpha * log_pi - q_new_actions).mean()


        """
        QF Loss
        """
        if self.backprop_q_loss_into_encoder:
            q1_pred = self.qf1(obs, actions, task_z_with_grad)
            q2_pred = self.qf2(obs, actions, task_z_with_grad)
        else:
            q1_pred = self.qf1(obs, actions, task_z_detached)
            q2_pred = self.qf2(obs, actions, task_z_detached)

        if not self.max_q_backup:
            target_q_values = torch.min(
                self.target_qf1(next_obs, new_next_actions, task_z_detached),
                self.target_qf2(next_obs, new_next_actions, task_z_detached),
            )

            if not self.deterministic_backup:
                target_q_values = target_q_values - alpha * new_log_pi
        else:
            """when using max q backup"""
            next_actions_temp, _ = self._get_policy_actions(
                next_obs, task_z_detached, num_actions=10,
            )
            target_qf1_values = (
                self._get_tensor_values(
                    next_obs, next_actions_temp, task_z_detached,
                    network=self.target_qf1).max(1)[0].view(-1, 1)
            )
            target_qf2_values = (
                self._get_tensor_values(
                    next_obs, next_actions_temp, task_z_detached,
                    network=self.target_qf2).max(1)[0].view(-1, 1)
            )
            target_q_values = torch.min(
                target_qf1_values, target_qf2_values)

        q_target = rewards + (1. - terminals) * self.discount * target_q_values
        q_target = q_target.detach()

        qf1_loss = self.qf_criterion(q1_pred, q_target)
        qf2_loss = self.qf_criterion(q2_pred, q_target)

        ## add CQL
        random_actions_tensor = ptu.zeros(
            q2_pred.shape[0] * self.num_random, actions.shape[-1]
        ).uniform_(-1, 1)
        curr_actions_tensor, curr_log_pis = self._get_policy_actions(
            obs, task_z_detached, num_actions=self.num_random)
        new_curr_actions_tensor, new_log_pis = self._get_policy_actions(
            next_obs, task_z_detached, num_actions=self.num_random)
        q1_rand = self._get_tensor_values(
            obs, random_actions_tensor, task_z_detached, network=self.qf1)
        q2_rand = self._get_tensor_values(
            obs, random_actions_tensor, task_z_detached, network=self.qf2)
        q1_curr_actions = self._get_tensor_values(
            obs, curr_actions_tensor, task_z_detached, network=self.qf1)
        q2_curr_actions = self._get_tensor_values(
            obs, curr_actions_tensor, task_z_detached, network=self.qf2)
        q1_next_actions = self._get_tensor_values(
            obs, new_curr_actions_tensor, task_z_detached, network=self.qf1)
        q2_next_actions = self._get_tensor_values(
            obs, new_curr_actions_tensor, task_z_detached, network=self.qf2)

        cat_q1 = torch.cat(
            [q1_rand, q1_pred.unsqueeze(1), q1_next_actions, q1_curr_actions], 1
        )
        cat_q2 = torch.cat(
            [q2_rand, q2_pred.unsqueeze(1), q2_next_actions, q2_curr_actions], 1
        )
        std_q1 = torch.std(cat_q1, dim=1)
        std_q2 = torch.std(cat_q2, dim=1)

        if self.min_q_version == 3:
            # importance sammpled version
            random_density = np.log(0.5 ** curr_actions_tensor.shape[-1])
            cat_q1 = torch.cat(
                [q1_rand - random_density,
                 q1_next_actions - new_log_pis.detach(),
                 q1_curr_actions - curr_log_pis.detach()], 1
            )
            cat_q2 = torch.cat(
                [q2_rand - random_density,
                 q2_next_actions - new_log_pis.detach(),
                 q2_curr_actions - curr_log_pis.detach()], 1
            )

        min_qf1_loss = torch.logsumexp(cat_q1 / self.temp,
                                       dim=1, ).mean() * self.min_q_weight * self.temp
        min_qf2_loss = torch.logsumexp(cat_q2 / self.temp,
                                       dim=1, ).mean() * self.min_q_weight * self.temp

        """Subtract the log likelihood of data"""
        min_qf1_loss = min_qf1_loss - q1_pred.mean() * self.min_q_weight
        min_qf2_loss = min_qf2_loss - q2_pred.mean() * self.min_q_weight

        if self.with_lagrange:
            alpha_prime = torch.clamp(self.log_alpha_prime.exp(), min=0.0,
                                      max=1000000.0)
            min_qf1_loss = alpha_prime * (min_qf1_loss - self.target_action_gap)
            min_qf2_loss = alpha_prime * (min_qf2_loss - self.target_action_gap)

            self.alpha_prime_optimizer.zero_grad()
            alpha_prime_loss = (-min_qf1_loss - min_qf2_loss) * 0.5
            alpha_prime_loss.backward(retain_graph=True)
            self.alpha_prime_optimizer.step()

        qf1_loss = qf1_loss + min_qf1_loss
        qf2_loss = qf2_loss + min_qf2_loss
        qf_loss = qf1_loss + qf2_loss

        """Context encoder/decoder"""
        kl_div = kl_divergence(p_z, self.agent.latent_prior).sum()
        kl_loss = self.kl_lambda * kl_div
        if self.train_context_decoder:
            # TODO: change to use a distribution
            reward_pred = self.context_decoder(obs, actions, task_z_with_grad)
            reward_prediction_loss = ((reward_pred - unscaled_rewards_flat)**2).mean()
            context_loss = kl_loss + reward_prediction_loss
        else:
            context_loss = kl_loss
            reward_prediction_loss = ptu.zeros(1)

        """
        Update networks
        """
        self.policy_optimizer.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.policy_optimizer.step()

        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        if self.train_encoder_decoder:
            self.context_optimizer.zero_grad()
        context_loss.backward(retain_graph=True)
        qf_loss.backward()
        self.qf1_optimizer.step()
        self.qf2_optimizer.step()
        if self.train_encoder_decoder:
            self.context_optimizer.step()
        """
        Soft Updates
        """
        ptu.soft_update_from_to(
            self.qf1, self.target_qf1, self.soft_target_tau
        )
        ptu.soft_update_from_to(
            self.qf2, self.target_qf2, self.soft_target_tau
        )
        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            This way, these statistics are only computed for one batch.
            """
            self.eval_statistics = OrderedDict()

            self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics['min QF1 Loss'] = np.mean(
                ptu.get_numpy(min_qf1_loss))
            self.eval_statistics['QF2 Loss'] = np.mean(
                ptu.get_numpy(qf2_loss))
            self.eval_statistics['min QF2 Loss'] = np.mean(
                ptu.get_numpy(min_qf2_loss))

            self.eval_statistics['Std QF1 values'] = np.mean(
                ptu.get_numpy(std_q1))
            self.eval_statistics['Std QF2 values'] = np.mean(
                ptu.get_numpy(std_q2))
            self.eval_statistics.update(create_stats_ordered_dict(
                'QF1 in-distribution values',
                ptu.get_numpy(q1_curr_actions),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'QF2 in-distribution values',
                ptu.get_numpy(q2_curr_actions),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'QF1 random values',
                ptu.get_numpy(q1_rand),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'QF2 random values',
                ptu.get_numpy(q2_rand),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'QF1 next_actions values',
                ptu.get_numpy(q1_next_actions),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'QF2 next_actions values',
                ptu.get_numpy(q2_next_actions),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'actions',
                ptu.get_numpy(actions)
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'rewards',
                ptu.get_numpy(rewards)
            ))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(q2_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            policy_mean = action_distrib.mean
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            policy_log_std = action_distrib.log_std
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))
            self.eval_statistics['task_embedding/kl_divergence'] = (
                ptu.get_numpy(kl_div)
            )
            self.eval_statistics['task_embedding/kl_loss'] = (
                ptu.get_numpy(kl_loss)
            )
            self.eval_statistics['task_embedding/reward_prediction_loss'] = (
                ptu.get_numpy(reward_prediction_loss)
            )
            self.eval_statistics['task_embedding/context_loss'] = (
                ptu.get_numpy(context_loss)
            )

            if self.use_automatic_entropy_tuning:
                self.eval_statistics['Alpha'] = alpha.item()
                self.eval_statistics['Alpha Loss'] = alpha_loss.item()

            if self.with_lagrange:
                self.eval_statistics['Alpha_prime'] = alpha_prime.item()
                self.eval_statistics['min_q1_loss'] = ptu.get_numpy(
                    min_qf1_loss).mean()
                self.eval_statistics['min_q2_loss'] = ptu.get_numpy(
                    min_qf2_loss).mean()
                self.eval_statistics[
                    'threshold action gap'] = self.target_action_gap
                self.eval_statistics[
                    'alpha prime loss'] = alpha_prime_loss.item()

    def _get_tensor_values(self, obs, actions, z, network=None):
        action_shape = actions.shape[0]
        batch_size = obs.shape[0]
        num_repeat = int(action_shape / batch_size)

        def stack(tensor):
            return tensor.unsqueeze(1).repeat(1, num_repeat, 1).view(
                batch_size * num_repeat, tensor.shape[1])

        obs_stacked = stack(obs)
        z_stacked = stack(z)

        preds = network(obs_stacked, actions, z_stacked)
        preds = preds.view(obs.shape[0], num_repeat, 1)
        return preds

    def _get_policy_actions(self, obs, z, num_actions):
        batch_size = obs.shape[0]

        def stack(tensor):
            return tensor.unsqueeze(1).repeat(1, num_actions, 1).view(batch_size * num_actions, tensor.shape[1])

        obs_stacked = stack(obs)
        z_stacked = stack(z)
        in_ = torch.cat([obs_stacked, z_stacked.detach()], dim=1)
        action_distrib = self.policy(in_)
        actions, log_pi = action_distrib.rsample_and_logprob()
        return actions, log_pi.view(batch_size, num_actions, 1)

    def get_snapshot(self):
        # NOTE: overriding parent method which also optionally saves the env
        snapshot = OrderedDict(
            qf1=self.qf1.state_dict(),
            qf2=self.qf2.state_dict(),
            target_qf1=self.target_qf1.state_dict(),
            target_qf2=self.target_qf2.state_dict(),
            policy=self.agent.policy.state_dict(),
            context_encoder=self.agent.context_encoder.state_dict(),
            context_decoder=self.context_decoder.state_dict(),
        )
        return snapshot

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    def get_diagnostics(self):
        stats = super().get_diagnostics()
        stats.update(self.eval_statistics)
        return stats

    ###### Torch stuff #####
    @property
    def networks(self):
        return [
            self.policy,
            self.qf1, self.qf2, self.target_qf1, self.target_qf2,
            self.context_encoder,
            self.context_decoder,
            self.reward_predictor,
        ]

    def training_mode(self, mode):
        for net in self.networks:
            net.train(mode)

    def to(self, device=None):
        if device == None:
            device = ptu.device
        for net in self.networks:
            net.to(device)
