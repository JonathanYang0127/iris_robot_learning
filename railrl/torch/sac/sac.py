"""
Soft actor-critic

Notes from Tuomas
a = tanh(z)
Q(a, s) = Q(tanh(z), s)
z is output of actor

Policy
pi ~ e^{Q(s, a)}
   ~ e^{Q(s, tanh(z))}

Mean
 - regularize gaussian means of policy towards zeros

Covariance
 - output log of diagonals
 - Clip the log of the diagonals
 - regularize the log towards zero
"""
from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

import railrl.torch.pytorch_util as ptu
from railrl.misc.eval_util import create_stats_ordered_dict
from railrl.torch.sac.policies import MakeDeterministic
from railrl.torch.torch_rl_algorithm import TorchRLAlgorithm


class SoftActorCritic(TorchRLAlgorithm):
    def __init__(
            self,
            env,
            policy,
            qf,
            vf,

            policy_lr=1e-3,
            qf_lr=1e-3,
            vf_lr=1e-3,
            policy_mean_reg_weight=1e-3,
            policy_std_reg_weight=1e-3,
            policy_pre_activation_weight=0.,
            optimizer_class=optim.Adam,

            train_policy_with_reparameterization=False,
            soft_target_tau=1e-2,
            target_update_period=1,
            eval_deterministic=True,
            target_hard_update_period=1,
            eval_policy=None,
            exploration_policy=None,
            use_automatic_entropy_tuning=False,
            target_entropy=None,
            **kwargs
    ):
        if eval_policy is None:
            if eval_deterministic:
                eval_policy = MakeDeterministic(policy)
            else:
                eval_policy = policy
        super().__init__(
            env=env,
            exploration_policy=exploration_policy or policy,
            eval_policy=eval_policy,
            **kwargs
        )
        self.policy = policy
        self.qf = qf
        self.vf = vf
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period
        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight
        self.policy_pre_activation_weight = policy_pre_activation_weight
        self.train_policy_with_reparameterization = (
            train_policy_with_reparameterization
        )
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy == None:
                self.target_entropy = -np.prod(self.env.action_space.shape).item() #heuristic value from Tuomas 
            else:
                self.target_entropy = target_entropy
            self.log_alpha = ptu.Variable(torch.zeros(1), requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )
        self.target_vf = vf.copy()
        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.qf_optimizer = optimizer_class(
            self.qf.parameters(),
            lr=qf_lr,
        )
        self.vf_optimizer = optimizer_class(
            self.vf.parameters(),
            lr=vf_lr,
        )

        self.target_hard_update_period = target_hard_update_period

    def _do_training(self):
        batch = self.get_batch()
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        q_pred = self.qf(obs, actions)
        v_pred = self.vf(obs)
        # Make sure policy accounts for squashing functions like tanh correctly!
        policy_outputs = self.policy(obs,
                                     reparameterize=self.train_policy_with_reparameterization,
                                     return_log_prob=True)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        """
        QF Loss
        """
        target_v_values = self.target_vf(next_obs)
        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_v_values
        qf_loss = self.qf_criterion(q_pred, q_target.detach())

        if self.use_automatic_entropy_tuning:
            """
            Alpha Loss
            """
            # alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            alpha_loss = ptu.Variable(torch.zeros(1), requires_grad=True)
            alpha = self.log_alpha.exp()

            """
            VF Loss
            """
            q_new_actions = self.qf(obs, new_actions)
            v_target = q_new_actions - alpha * log_pi
            vf_loss = self.vf_criterion(v_pred, v_target.detach())

            """
            Policy Loss
            """
            if self.train_policy_with_reparameterization:
                policy_loss = (alpha * log_pi - q_new_actions).mean()
            else:
                log_policy_target = q_new_actions - v_pred
                policy_loss = (
                        log_pi * (alpha * log_pi - log_policy_target).detach()
                ).mean()
        else:
            """
            VF Loss
            """
            q_new_actions = self.qf(obs, new_actions)
            v_target = q_new_actions - log_pi
            vf_loss = self.vf_criterion(v_pred, v_target.detach())

            """
            Policy Loss
            """
            if self.train_policy_with_reparameterization:
                policy_loss = (log_pi - q_new_actions).mean()
            else:
                log_policy_target = q_new_actions - v_pred
                policy_loss = (
                    log_pi * (log_pi - log_policy_target).detach()
                ).mean()
        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean**2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std**2).mean()
        pre_tanh_value = policy_outputs[-1]
        pre_activation_reg_loss = self.policy_pre_activation_weight * (
            (pre_tanh_value**2).sum(dim=1).mean()
        )
        policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
        policy_loss = policy_loss + policy_reg_loss

        """
        Update networks
        """
        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()

        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        if self.use_automatic_entropy_tuning:
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()


        if self._n_train_steps_total % self.target_update_period == 0:
            self._update_target_network()

        """
        Save some statistics for eval
        """
        if self.need_to_update_eval_statistics:
            self.need_to_update_eval_statistics = False
            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics['VF Loss'] = np.mean(ptu.get_numpy(vf_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Predictions',
                ptu.get_numpy(q_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'V Predictions',
                ptu.get_numpy(v_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))
            if self.use_automatic_entropy_tuning:
                self.eval_statistics['Alpha'] = ptu.get_numpy(alpha)[0]
                self.eval_statistics['Alpha Loss'] = ptu.get_numpy(alpha_loss)[0]

    @property
    def networks(self):
        return [
            self.policy,
            self.qf,
            self.vf,
            self.target_vf,
        ]

    def offline_evaluate(self, epoch):
        raise NotImplementedError()

    def _update_target_network(self):
        ptu.soft_update_from_to(self.vf, self.target_vf, self.soft_target_tau)

    def get_epoch_snapshot(self, epoch):
        snapshot = super().get_epoch_snapshot(epoch)
        snapshot['qf'] = self.qf
        snapshot['policy'] = self.policy
        snapshot['vf'] = self.vf
        snapshot['target_vf'] = self.target_vf
        return snapshot
