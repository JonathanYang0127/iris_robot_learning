import pickle
from collections import OrderedDict
import numpy as np
import torch
import torch.optim as optim
from rlkit.torch.sac.policies import MakeDeterministic
from torch import nn as nn
import rlkit.torch.pytorch_util as ptu
from rlkit.misc.eval_util import create_stats_ordered_dict
from rlkit.torch.core import np_to_pytorch_batch
from rlkit.torch.torch_rl_algorithm import TorchTrainer
from rlkit.core import logger
from rlkit.core.logging import add_prefix
from rlkit.misc.ml_util import PiecewiseLinearSchedule, ConstantSchedule
import torch.nn.functional as F
from rlkit.torch.networks import LinearTransform
import time
from collections import OrderedDict
import copy
import numpy as np

import torch
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu

from rlkit.core.meta_rl_algorithm import MetaRLAlgorithm
from rlkit.misc.eval_util import create_stats_ordered_dict
from rlkit.torch.core import np_to_pytorch_batch


class PearlTrainer(MetaRLAlgorithm):
    def __init__(
            self,
            env,
            train_tasks,
            eval_tasks,
            latent_dim,
            nets,

            # AWAC Params
            discount=0.99,
            reward_scale=1.0,
            beta=1.0,
            beta_schedule_kwargs=None,

            policy_lr=1e-3,
            qf_lr=1e-3,
            policy_weight_decay=0,
            q_weight_decay=0,
            optimizer_class=optim.Adam,

            soft_target_tau=1e-2,
            target_update_period=1,
            plotter=None,
            render_eval_paths=False,

            use_automatic_entropy_tuning=True,
            target_entropy=None,

            bc_num_pretrain_steps=0,
            q_num_pretrain1_steps=0,
            q_num_pretrain2_steps=0,
            bc_batch_size=128,
            alpha=1.0,

            policy_update_period=1,
            q_update_period=1,

            weight_loss=True,
            compute_bc=True,
            use_awr_update=True,
            use_reparam_update=False,

            bc_weight=0.0,
            rl_weight=1.0,
            reparam_weight=1.0,
            awr_weight=1.0,

            post_pretrain_hyperparams=None,
            post_bc_pretrain_hyperparams=None,

            awr_use_mle_for_vf=False,
            vf_K=1,
            awr_sample_actions=False,
            buffer_policy_sample_actions=False,
            awr_min_q=False,
            brac=False,

            reward_transform_class=None,
            reward_transform_kwargs=None,
            terminal_transform_class=None,
            terminal_transform_kwargs=None,

            pretraining_logging_period=1000,

            train_bc_on_rl_buffer=False,
            use_automatic_beta_tuning=False,
            beta_epsilon=1e-10,
            normalize_over_batch=True,
            normalize_over_state="advantage",
            Z_K=10,
            clip_score=None,
            validation_qlearning=False,

            mask_positive_advantage=False,
            buffer_policy_reset_period=-1,
            num_buffer_policy_train_steps_on_reset=100,
            advantage_weighted_buffer_loss=True,

            # PEARL Params
            policy_lr=1e-3,
            qf_lr=1e-3,
            vf_lr=1e-3,
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
            back_prop_reward_prediction_into_encoder=False,

            train_reward_pred_in_unsupervised_phase=False,
            use_encoder_snapshot_for_reward_pred_in_unsupervised_phase=False,

            soft_target_tau=1e-2,
            plotter=None,
            render_eval_paths=False,
            **kwargs
    ):
        super().__init__(
            env=env,
            agent=nets[0],
            train_tasks=train_tasks,
            eval_tasks=eval_tasks,
            **kwargs
        )

        self.soft_target_tau = soft_target_tau
        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight
        self.policy_pre_activation_weight = policy_pre_activation_weight
        self.plotter = plotter
        self.render_eval_paths = render_eval_paths
        self.back_prop_reward_prediction_into_encoder = back_prop_reward_prediction_into_encoder

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

        self.qf1, self.qf2, self.vf = nets[1:]
        self.target_vf = copy.deepcopy(self.vf)

        self.policy_optimizer = optimizer_class(
            self.agent.policy.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )
        self.vf_optimizer = optimizer_class(
            self.vf.parameters(),
            lr=vf_lr,
        )
        self.context_optimizer = optimizer_class(
            self.agent.context_encoder.parameters(),
            lr=context_lr,
        )
        self.reward_predictor_optimizer = optimizer_class(
            self.agent.reward_predictor.parameters(),
            lr=context_lr,
        )

        # AWAC params
        self.use_awr_update = use_awr_update
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(self.env.action_space.shape).item()  # heuristic value from Tuomas
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )

        self.awr_use_mle_for_vf = awr_use_mle_for_vf
        self.vf_K = vf_K
        self.awr_sample_actions = awr_sample_actions
        self.awr_min_q = awr_min_q

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        self.optimizers = {}

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            weight_decay=policy_weight_decay,
            lr=policy_lr,
        )
        self.optimizers[self.policy] = self.policy_optimizer
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            weight_decay=q_weight_decay,
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            weight_decay=q_weight_decay,
            lr=qf_lr,
        )

        if buffer_policy and train_bc_on_rl_buffer:
            self.buffer_policy_optimizer =  optimizer_class(
                self.buffer_policy.parameters(),
                weight_decay=policy_weight_decay,
                lr=policy_lr,
            )
            self.optimizers[self.buffer_policy] = self.buffer_policy_optimizer
            self.optimizer_class=optimizer_class
            self.policy_weight_decay=policy_weight_decay
            self.policy_lr = policy_lr

        self.use_automatic_beta_tuning = use_automatic_beta_tuning and buffer_policy and train_bc_on_rl_buffer
        self.beta_epsilon=beta_epsilon
        if self.use_automatic_beta_tuning:
            self.log_beta = ptu.zeros(1, requires_grad=True)
            self.beta_optimizer = optimizer_class(
                [self.log_beta],
                lr=policy_lr,
            )
        else:
            self.beta = beta
            self.beta_schedule_kwargs = beta_schedule_kwargs
            if beta_schedule_kwargs is None:
                self.beta_schedule = ConstantSchedule(beta)
            else:
                schedule_class = beta_schedule_kwargs.pop("schedule_class", PiecewiseLinearSchedule)
                self.beta_schedule = schedule_class(**beta_schedule_kwargs)

        self.discount = discount
        self.reward_scale = reward_scale
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

        self.bc_num_pretrain_steps = bc_num_pretrain_steps
        self.q_num_pretrain1_steps = q_num_pretrain1_steps
        self.q_num_pretrain2_steps = q_num_pretrain2_steps
        self.bc_batch_size = bc_batch_size
        self.rl_weight = rl_weight
        self.bc_weight = bc_weight
        self.eval_policy = MakeDeterministic(self.policy)
        self.compute_bc = compute_bc
        self.alpha = alpha
        self.q_update_period = q_update_period
        self.policy_update_period = policy_update_period
        self.weight_loss = weight_loss

        self.reparam_weight = reparam_weight
        self.awr_weight = awr_weight
        self.post_pretrain_hyperparams = post_pretrain_hyperparams
        self.post_bc_pretrain_hyperparams = post_bc_pretrain_hyperparams
        self.update_policy = True
        self.pretraining_logging_period = pretraining_logging_period
        self.normalize_over_batch = normalize_over_batch
        self.normalize_over_state = normalize_over_state
        self.Z_K = Z_K

        self.reward_transform_class = reward_transform_class or LinearTransform
        self.reward_transform_kwargs = reward_transform_kwargs or dict(m=1, b=0)
        self.terminal_transform_class = terminal_transform_class or LinearTransform
        self.terminal_transform_kwargs = terminal_transform_kwargs or dict(m=1, b=0)
        self.reward_transform = self.reward_transform_class(**self.reward_transform_kwargs)
        self.terminal_transform = self.terminal_transform_class(**self.terminal_transform_kwargs)
        self.use_reparam_update = use_reparam_update
        self.clip_score = clip_score
        self.buffer_policy_sample_actions = buffer_policy_sample_actions

        self.train_bc_on_rl_buffer = train_bc_on_rl_buffer and buffer_policy
        self.validation_qlearning = validation_qlearning
        self.brac = brac
        self.mask_positive_advantage = mask_positive_advantage
        self.buffer_policy_reset_period = buffer_policy_reset_period
        self.num_buffer_policy_train_steps_on_reset=num_buffer_policy_train_steps_on_reset
        self.advantage_weighted_buffer_loss=advantage_weighted_buffer_loss

    ###### Torch stuff #####
    @property
    def networks(self):
        return self.agent.networks + [self.agent] + [self.qf1, self.qf2, self.vf, self.target_vf]

    def training_mode(self, mode):
        for net in self.networks:
            net.train(mode)

    def to(self, device=None):
        if device == None:
            device = ptu.device
        for net in self.networks:
            net.to(device)

    ##### Data handling #####
    def unpack_batch(self, batch, sparse_reward=False):
        ''' unpack a batch and return individual elements '''
        o = batch['observations'][None, ...]
        a = batch['actions'][None, ...]
        if sparse_reward:
            r = batch['sparse_rewards'][None, ...]
        else:
            r = batch['rewards'][None, ...]
        no = batch['next_observations'][None, ...]
        t = batch['terminals'][None, ...]
        return [o, a, r, no, t]

    def sample_sac(self, indices):
        ''' sample batch of training data from a list of tasks for training the actor-critic '''
        # this batch consists of transitions sampled randomly from replay buffer
        # rewards are always dense
        batches = [np_to_pytorch_batch(self.replay_buffer.random_batch(idx, batch_size=self.batch_size)) for idx in indices]
        unpacked = [self.unpack_batch(batch) for batch in batches]
        # group like elements together
        unpacked = [[x[i] for x in unpacked] for i in range(len(unpacked[0]))]
        unpacked = [torch.cat(x, dim=0) for x in unpacked]
        return unpacked

    def sample_context(self, indices, also_return_dict=False):
        ''' sample batch of context from a list of tasks from the replay buffer '''
        # make method work given a single task index
        if not hasattr(indices, '__iter__'):
            indices = [indices]
        batches = [np_to_pytorch_batch(self.enc_replay_buffer.random_batch(idx, batch_size=self.embedding_batch_size, sequence=self.recurrent)) for idx in indices]
        env_info_sizes=dict(),
        context = [self.unpack_batch(batch, sparse_reward=self.sparse_rewards) for batch in batches]
        # group like elements together
        context = [[x[i] for x in context] for i in range(len(context[0]))]
        context = [torch.cat(x, dim=0) for x in context]
        # full context consists of [obs, act, rewards, next_obs, terms]
        # if dynamics don't change across tasks, don't include next_obs
        # don't include terminals in context
        if self.use_next_obs_in_context:
            context = torch.cat(context[:-1], dim=2)
        else:
            context = torch.cat(context[:-2], dim=2)

        if also_return_dict:
            context_dict = {}
            for k in batches[0]:
                context_dict[k] = torch.stack([d[k] for d in batches], dim=0)
            return context, context_dict
        else:
            return context

    ##### Training #####
    def _do_training(self, indices):
        mb_size = self.embedding_mini_batch_size
        num_updates = self.embedding_batch_size // mb_size

        # sample context batch
        context_batch, context_dict = self.sample_context(indices, also_return_dict=True)

        # zero out context and hidden encoder state
        self.agent.clear_z(num_tasks=len(indices))

        # do this in a loop so we can truncate backprop in the recurrent encoder
        for i in range(num_updates):
            context = context_batch[:, i * mb_size: i * mb_size + mb_size, :]
            minibatch_context_dict = {}
            for k in ['rewards', 'observations', 'actions']:
                minibatch_context_dict[k] = context_dict[k][:, i * mb_size: i * mb_size + mb_size, :]
            self._take_step(indices, context, minibatch_context_dict)

            # stop backprop
            self.agent.detach_z()

    def _min_q(self, obs, actions, task_z):
        q1 = self.qf1(obs, actions, task_z.detach())
        q2 = self.qf2(obs, actions, task_z.detach())
        min_q = torch.min(q1, q2)
        return min_q

    def _update_target_network(self):
        ptu.soft_update_from_to(self.vf, self.target_vf, self.soft_target_tau)

    def _take_step(self, indices, context, context_dict):

        num_tasks = len(indices)

        # data is (task, batch, feat)
        obs, actions, rewards, next_obs, terms = self.sample_sac(indices)

        # run inference in networks
        action_distrib, task_z = self.agent(obs, context, return_task_z=True)
        new_actions, log_pi, pre_tanh_value = action_distrib.rsample_and_logprob(return_pre_tanh_value=True)
        policy_mean = action_distrib.mean
        policy_log_std = action_distrib.log_std

        # flattens out the task dimension
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)

        # Q and V networks
        # encoder will only get gradients from Q nets
        q1_pred = self.qf1(obs, actions, task_z)
        q2_pred = self.qf2(obs, actions, task_z)
        v_pred = self.vf(obs, task_z.detach())
        # get targets for use in V and Q updates
        with torch.no_grad():
            target_v_values = self.target_vf(next_obs, task_z)

        # KL constraint on z if probabilistic
        self.context_optimizer.zero_grad()
        if self.use_information_bottleneck:
            kl_div = self.agent.compute_kl_div()
            kl_loss = self.kl_lambda * kl_div
            kl_loss.backward(retain_graph=True)

        # reward prediction loop
        if not self.in_unsupervised_phase or self.train_reward_pred_in_unsupervised_phase:
            t_rp, b_rp, _ = context_dict['observations'].shape
            obs_rp = context_dict['observations'].view(t_rp * b_rp, -1)
            actions_rp = context_dict['actions'].view(t_rp * b_rp, -1)
            _, task_z_rp = self.agent(context_dict['observations'], context, return_task_z=True)
            if self.back_prop_reward_prediction_into_encoder:
                reward_pred = self.agent.reward_predictor(obs_rp, actions_rp, task_z_rp)
            else:
                reward_pred = self.agent.reward_predictor(obs_rp, actions_rp, task_z_rp.detach())

            rewards_rp = context_dict['rewards'].view(t_rp * b_rp, -1)
            reward_pred_loss = self.reward_pred_criterion(reward_pred, rewards_rp)
            self.reward_predictor_optimizer.zero_grad()

        # qf and encoder update (note encoder does not get grads from policy or vf)
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        rewards_flat = rewards.view(self.batch_size * num_tasks, -1)
        # scale rewards for Bellman update
        rewards_flat = rewards_flat * self.reward_scale
        terms_flat = terms.view(self.batch_size * num_tasks, -1)
        q_target = rewards_flat + (1. - terms_flat) * self.discount * target_v_values
        qf_loss = torch.mean((q1_pred - q_target) ** 2) + torch.mean((q2_pred - q_target) ** 2)
        if not self.in_unsupervised_phase or self.train_reward_pred_in_unsupervised_phase:
            reward_pred_loss.backward()
        qf_loss.backward()
        self.qf1_optimizer.step()
        self.qf2_optimizer.step()
        self.context_optimizer.step()
        if not self.in_unsupervised_phase or self.train_reward_pred_in_unsupervised_phase:
            self.reward_predictor_optimizer.step()

        # compute min Q on the new actions
        min_q_new_actions = self._min_q(obs, new_actions, task_z)

        # vf update
        v_target = min_q_new_actions - log_pi
        vf_loss = self.vf_criterion(v_pred, v_target.detach())
        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()
        self._update_target_network()

        # policy update
        # n.b. policy update includes dQ/da
        log_policy_target = min_q_new_actions

        policy_loss = (
                log_pi - log_policy_target
        ).mean()

        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean**2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std**2).mean()
        pre_activation_reg_loss = self.policy_pre_activation_weight * (
            (pre_tanh_value**2).sum(dim=1).mean()
        )
        policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
        policy_loss = policy_loss + policy_reg_loss

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # save some statistics for eval
        if self.eval_statistics is None:
            # eval should set this to None.
            # this way, these statistics are only computed for one batch.
            self.eval_statistics = OrderedDict()
            if self.use_information_bottleneck:
                z_mean = np.mean(np.abs(ptu.get_numpy(self.agent.z_means[0])))
                z_sig = np.mean(ptu.get_numpy(self.agent.z_vars[0]))
                self.eval_statistics['Z mean train'] = z_mean
                self.eval_statistics['Z variance train'] = z_sig
                self.eval_statistics['KL Divergence'] = ptu.get_numpy(kl_div)
                self.eval_statistics['KL Loss'] = ptu.get_numpy(kl_loss)

            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics['VF Loss'] = np.mean(ptu.get_numpy(vf_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            if not self.in_unsupervised_phase or self.train_reward_pred_in_unsupervised_phase:
                self.eval_statistics['Reward Prediction Loss'] = np.mean(
                    ptu.get_numpy(reward_pred_loss))
            else:
                self.eval_statistics['Reward Prediction Loss'] = 0
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Predictions',
                ptu.get_numpy(q1_pred),
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

        # AWAC params

        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        weights = batch.get('weights', None)
        if self.reward_transform:
            rewards = self.reward_transform(rewards)

        if self.terminal_transform:
            terminals = self.terminal_transform(terminals)
        """
        Policy and Alpha Loss
        """
        dist = self.policy(obs)
        new_obs_actions, log_pi = dist.rsample_and_logprob()
        policy_mle = dist.mle_estimate()

        if self.brac:
            buf_dist = self.buffer_policy(obs)
            buf_log_pi = buf_dist.log_prob(actions)
            rewards = rewards + buf_log_pi

        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = self.alpha

        """
        QF Loss
        """
        q1_pred = self.qf1(obs, actions)
        q2_pred = self.qf2(obs, actions)
        # Make sure policy accounts for squashing functions like tanh correctly!
        next_dist = self.policy(next_obs)
        new_next_actions, new_log_pi = next_dist.rsample_and_logprob()
        target_q_values = torch.min(
            self.target_qf1(next_obs, new_next_actions),
            self.target_qf2(next_obs, new_next_actions),
        ) - alpha * new_log_pi

        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
        qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
        qf2_loss = self.qf_criterion(q2_pred, q_target.detach())

        """
        Policy Loss
        """
        qf1_new_actions = self.qf1(obs, new_obs_actions)
        qf2_new_actions = self.qf2(obs, new_obs_actions)
        q_new_actions = torch.min(
            qf1_new_actions,
            qf2_new_actions,
        )

        # Advantage-weighted regression
        if self.awr_use_mle_for_vf:
            v1_pi = self.qf1(obs, policy_mle)
            v2_pi = self.qf2(obs, policy_mle)
            v_pi = torch.min(v1_pi, v2_pi)
        else:
            if self.vf_K > 1:
                vs = []
                for i in range(self.vf_K):
                    u = dist.sample()
                    q1 = self.qf1(obs, u)
                    q2 = self.qf2(obs, u)
                    v = torch.min(q1, q2)
                    # v = q1
                    vs.append(v)
                v_pi = torch.cat(vs, 1).mean(dim=1)
            else:
                # v_pi = self.qf1(obs, new_obs_actions)
                v1_pi = self.qf1(obs, new_obs_actions)
                v2_pi = self.qf2(obs, new_obs_actions)
                v_pi = torch.min(v1_pi, v2_pi)

        if self.awr_sample_actions:
            u = new_obs_actions
            if self.awr_min_q:
                q_adv = q_new_actions
            else:
                q_adv = qf1_new_actions
        elif self.buffer_policy_sample_actions:
            buf_dist = self.buffer_policy(obs)
            u, _ = buf_dist.rsample_and_logprob()
            qf1_buffer_actions = self.qf1(obs, u)
            qf2_buffer_actions = self.qf2(obs, u)
            q_buffer_actions = torch.min(
                qf1_buffer_actions,
                qf2_buffer_actions,
            )
            if self.awr_min_q:
                q_adv = q_buffer_actions
            else:
                q_adv = qf1_buffer_actions
        else:
            u = actions
            if self.awr_min_q:
                q_adv = torch.min(q1_pred, q2_pred)
            else:
                q_adv = q1_pred

        policy_logpp = dist.log_prob(u)

        if self.use_automatic_beta_tuning:
            buffer_dist = self.buffer_policy(obs)
            beta = self.log_beta.exp()
            kldiv = torch.distributions.kl.kl_divergence(dist, buffer_dist)
            beta_loss = -1*(beta*(kldiv-self.beta_epsilon).detach()).mean()

            self.beta_optimizer.zero_grad()
            beta_loss.backward()
            self.beta_optimizer.step()
        else:
            beta = self.beta_schedule.get_value(self._n_train_steps_total)

        if self.normalize_over_state == "advantage":
            score = q_adv - v_pi
            if self.mask_positive_advantage:
                score = torch.sign(score)
        elif self.normalize_over_state == "Z":
            buffer_dist = self.buffer_policy(obs)
            K = self.Z_K
            buffer_obs = []
            buffer_actions = []
            log_bs = []
            log_pis = []
            for i in range(K):
                u = buffer_dist.sample()
                log_b = buffer_dist.log_prob(u)
                log_pi = dist.log_prob(u)
                buffer_obs.append(obs)
                buffer_actions.append(u)
                log_bs.append(log_b)
                log_pis.append(log_pi)
            buffer_obs = torch.cat(buffer_obs, 0)
            buffer_actions = torch.cat(buffer_actions, 0)
            p_buffer = torch.exp(torch.cat(log_bs, 0).sum(dim=1, ))
            log_pi = torch.cat(log_pis, 0)
            log_pi = log_pi.sum(dim=1, )
            q1_b = self.qf1(buffer_obs, buffer_actions)
            q2_b = self.qf2(buffer_obs, buffer_actions)
            q_b = torch.min(q1_b, q2_b)
            q_b = torch.reshape(q_b, (-1, K))
            adv_b = q_b - v_pi
            # if self._n_train_steps_total % 100 == 0:
            #     import ipdb; ipdb.set_trace()
            # Z = torch.exp(adv_b / beta).mean(dim=1, keepdim=True)
            # score = torch.exp((q_adv - v_pi) / beta) / Z
            # score = score / sum(score)
            logK = torch.log(ptu.tensor(float(K)))
            logZ = torch.logsumexp(adv_b/beta - logK, dim=1, keepdim=True)
            logS = (q_adv - v_pi)/beta - logZ
            # logZ = torch.logsumexp(q_b/beta - logK, dim=1, keepdim=True)
            # logS = q_adv/beta - logZ
            score = F.softmax(logS, dim=0) # score / sum(score)
        else:
            error

        if self.clip_score is not None:
            score = torch.clamp(score, max=self.clip_score)

        if self.weight_loss and weights is None:
            if self.normalize_over_batch == True:
                weights = F.softmax(score / beta, dim=0)
            elif self.normalize_over_batch == "whiten":
                adv_mean = torch.mean(score)
                adv_std = torch.std(score) + 1e-5
                normalized_score = (score - adv_mean) / adv_std
                weights = torch.exp(normalized_score / beta)
            elif self.normalize_over_batch == "exp":
                weights = torch.exp(score / beta)
            elif self.normalize_over_batch == "step_fn":
                weights = (score > 0).float()
            elif self.normalize_over_batch == False:
                weights = score
            else:
                error
        weights = weights[:, 0]

        policy_loss = alpha * log_pi.mean()

        if self.use_awr_update and self.weight_loss:
            policy_loss = policy_loss + self.awr_weight * (-policy_logpp * len(weights)*weights.detach()).mean()
        elif self.use_awr_update:
            policy_loss = policy_loss + self.awr_weight * (-policy_logpp).mean()

        if self.use_reparam_update:
            policy_loss = policy_loss + self.reparam_weight * (-q_new_actions).mean()

        policy_loss = self.rl_weight * policy_loss
        if self.compute_bc:
            train_policy_loss, train_logp_loss, train_mse_loss, _ = self.run_bc_batch(self.demo_train_buffer, self.policy)
            policy_loss = policy_loss + self.bc_weight * train_policy_loss



        if not pretrain and self.buffer_policy_reset_period > 0 and self._n_train_steps_total % self.buffer_policy_reset_period==0:
            del self.buffer_policy_optimizer
            self.buffer_policy_optimizer =  self.optimizer_class(
                self.buffer_policy.parameters(),
                weight_decay=self.policy_weight_decay,
                lr=self.policy_lr,
            )
            self.optimizers[self.buffer_policy] = self.buffer_policy_optimizer
            for i in range(self.num_buffer_policy_train_steps_on_reset):
                if self.train_bc_on_rl_buffer:
                    if self.advantage_weighted_buffer_loss:
                        buffer_dist = self.buffer_policy(obs)
                        buffer_u = actions
                        buffer_new_obs_actions, _ = buffer_dist.rsample_and_logprob()
                        buffer_policy_logpp = buffer_dist.log_prob(buffer_u)
                        buffer_policy_logpp = buffer_policy_logpp[:, None]

                        buffer_q1_pred = self.qf1(obs, buffer_u)
                        buffer_q2_pred = self.qf2(obs, buffer_u)
                        buffer_q_adv = torch.min(buffer_q1_pred, buffer_q2_pred)

                        buffer_v1_pi = self.qf1(obs, buffer_new_obs_actions)
                        buffer_v2_pi = self.qf2(obs, buffer_new_obs_actions)
                        buffer_v_pi = torch.min(buffer_v1_pi, buffer_v2_pi)

                        buffer_score = buffer_q_adv - buffer_v_pi
                        buffer_weights = F.softmax(buffer_score / beta, dim=0)
                        buffer_policy_loss = self.awr_weight * (-buffer_policy_logpp * len(buffer_weights)*buffer_weights.detach()).mean()
                    else:
                        buffer_policy_loss, buffer_train_logp_loss, buffer_train_mse_loss, _ = self.run_bc_batch(
                        self.replay_buffer.train_replay_buffer, self.buffer_policy)

                    self.buffer_policy_optimizer.zero_grad()
                    buffer_policy_loss.backward(retain_graph=True)
                    self.buffer_policy_optimizer.step()

        if self.train_bc_on_rl_buffer:
            if self.advantage_weighted_buffer_loss:
                buffer_dist = self.buffer_policy(obs)
                buffer_u = actions
                buffer_new_obs_actions, _ = buffer_dist.rsample_and_logprob()
                buffer_policy_logpp = buffer_dist.log_prob(buffer_u)
                buffer_policy_logpp = buffer_policy_logpp[:, None]

                buffer_q1_pred = self.qf1(obs, buffer_u)
                buffer_q2_pred = self.qf2(obs, buffer_u)
                buffer_q_adv = torch.min(buffer_q1_pred, buffer_q2_pred)

                buffer_v1_pi = self.qf1(obs, buffer_new_obs_actions)
                buffer_v2_pi = self.qf2(obs, buffer_new_obs_actions)
                buffer_v_pi = torch.min(buffer_v1_pi, buffer_v2_pi)

                buffer_score = buffer_q_adv - buffer_v_pi
                buffer_weights = F.softmax(buffer_score / beta, dim=0)
                buffer_policy_loss = self.awr_weight * (-buffer_policy_logpp * len(buffer_weights)*buffer_weights.detach()).mean()
            else:
                buffer_policy_loss, buffer_train_logp_loss, buffer_train_mse_loss, _ = self.run_bc_batch(
                    self.replay_buffer.train_replay_buffer, self.buffer_policy)



        """
        Update networks
        """
        if self._n_train_steps_total % self.q_update_period == 0:
            self.qf1_optimizer.zero_grad()
            qf1_loss.backward()
            self.qf1_optimizer.step()

            self.qf2_optimizer.zero_grad()
            qf2_loss.backward()
            self.qf2_optimizer.step()

        if self._n_train_steps_total % self.policy_update_period == 0 and self.update_policy:
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

        if self.train_bc_on_rl_buffer and self._n_train_steps_total % self.policy_update_period == 0 :
            self.buffer_policy_optimizer.zero_grad()
            buffer_policy_loss.backward()
            self.buffer_policy_optimizer.step()



        """
        Soft Updates
        """
        if self._n_train_steps_total % self.target_update_period == 0:
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
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            policy_loss = (log_pi - q_new_actions).mean()

            self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
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
            self.eval_statistics.update(create_stats_ordered_dict(
                'rewards',
                ptu.get_numpy(rewards),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'terminals',
                ptu.get_numpy(terminals),
            ))
            policy_statistics = add_prefix(dist.get_diagnostics(), "policy/")
            self.eval_statistics.update(policy_statistics)
            self.eval_statistics.update(create_stats_ordered_dict(
                'Advantage Weights',
                ptu.get_numpy(weights),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Advantage Score',
                ptu.get_numpy(score),
            ))

            if self.normalize_over_state == "Z":
                self.eval_statistics.update(create_stats_ordered_dict(
                    'logZ',
                    ptu.get_numpy(logZ),
                ))

            if self.use_automatic_entropy_tuning:
                self.eval_statistics['Alpha'] = alpha.item()
                self.eval_statistics['Alpha Loss'] = alpha_loss.item()

            if self.compute_bc:
                test_policy_loss, test_logp_loss, test_mse_loss, _ = self.run_bc_batch(self.demo_test_buffer, self.policy)
                self.eval_statistics.update({
                    "bc/Train Logprob Loss": ptu.get_numpy(train_logp_loss),
                    "bc/Test Logprob Loss": ptu.get_numpy(test_logp_loss),
                    "bc/Train MSE": ptu.get_numpy(train_mse_loss),
                    "bc/Test MSE": ptu.get_numpy(test_mse_loss),
                    "bc/train_policy_loss": ptu.get_numpy(train_policy_loss),
                    "bc/test_policy_loss": ptu.get_numpy(test_policy_loss),
                })
            if self.train_bc_on_rl_buffer:
                _, buffer_train_logp_loss, _, _ = self.run_bc_batch(
                    self.replay_buffer.train_replay_buffer,
                    self.buffer_policy)

                _, buffer_test_logp_loss, _, _ = self.run_bc_batch(
                    self.replay_buffer.validation_replay_buffer,
                    self.buffer_policy)
                buffer_dist = self.buffer_policy(obs)
                kldiv = torch.distributions.kl.kl_divergence(dist, buffer_dist)

                _, train_offline_logp_loss, _, _ = self.run_bc_batch(
                    self.demo_train_buffer,
                    self.buffer_policy)

                _, test_offline_logp_loss, _, _ = self.run_bc_batch(
                    self.demo_test_buffer,
                    self.buffer_policy)

                self.eval_statistics.update({

                    "buffer_policy/Train Online Logprob": -1 * ptu.get_numpy(buffer_train_logp_loss),
                    "buffer_policy/Test Online Logprob": -1 * ptu.get_numpy(buffer_test_logp_loss),

                    "buffer_policy/Train Offline Logprob": -1 * ptu.get_numpy(train_offline_logp_loss),
                    "buffer_policy/Test Offline Logprob": -1 * ptu.get_numpy(test_offline_logp_loss),

                    "buffer_policy/train_policy_loss": ptu.get_numpy(buffer_policy_loss),
                    # "buffer_policy/test_policy_loss": ptu.get_numpy(buffer_test_policy_loss),
                    "buffer_policy/kl_div": ptu.get_numpy(kldiv.mean()),
                })
            if self.use_automatic_beta_tuning:
                self.eval_statistics.update({
                    "adaptive_beta/beta":ptu.get_numpy(beta.mean()),
                    "adaptive_beta/beta loss": ptu.get_numpy(beta_loss.mean()),
                })

            if self.validation_qlearning:
                train_data = self.replay_buffer.validation_replay_buffer.random_batch(self.bc_batch_size)
                train_data = np_to_pytorch_batch(train_data)
                obs = train_data['observations']
                next_obs = train_data['next_observations']
                # goals = train_data['resampled_goals']
                train_data['observations'] = obs # torch.cat((obs, goals), dim=1)
                train_data['next_observations'] = next_obs # torch.cat((next_obs, goals), dim=1)
                self.test_from_torch(train_data)

        self._n_train_steps_total += 1

    def get_epoch_snapshot(self, epoch):
        # NOTE: overriding parent method which also optionally saves the env
        snapshot = OrderedDict(
            qf1=self.qf1.state_dict(),
            qf2=self.qf2.state_dict(),
            policy=self.agent.policy.state_dict(),
            vf=self.vf.state_dict(),
            target_vf=self.target_vf.state_dict(),
            context_encoder=self.agent.context_encoder.state_dict(),
            buffer_policy=self.buffer_policy,
        )
        return snapshot
