import pickle
from collections import OrderedDict
import numpy as np
import torch
import torch.optim as optim
from railrl.torch.sac.policies import MakeDeterministic
from torch import nn as nn
import railrl.torch.pytorch_util as ptu
from railrl.misc.eval_util import create_stats_ordered_dict
from railrl.torch.core import np_to_pytorch_batch
from railrl.torch.torch_rl_algorithm import TorchTrainer
from railrl.core import logger
from railrl.core.logging import add_prefix
from railrl.misc.ml_util import PiecewiseLinearSchedule, ConstantSchedule
import torch.nn.functional as F
from railrl.torch.networks import LinearTransform
import time

class AWRSACTrainer(TorchTrainer):
    def __init__(
            self,
            env,
            policy,
            qf1,
            qf2,
            target_qf1,
            target_qf2,
            buffer_policy=None,

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
            bc_loss_type="mle",
            awr_loss_type="mle",
            save_bc_policies=0,
            alpha=1.0,

            policy_update_period=1,
            q_update_period=1,

            weight_loss=True,
            compute_bc=True,

            bc_weight=0.0,
            rl_weight=1.0,
            use_awr_update=True,
            use_reparam_update=False,
            use_klac_update=False,
            reparam_weight=1.0,
            awr_weight=1.0,
            klac_weight=1.0,
            klac_K=10,
            post_pretrain_hyperparams=None,
            post_bc_pretrain_hyperparams=None,

            awr_use_mle_for_vf=False,
            vf_K=1,
            awr_sample_actions=False,
            awr_min_q=False,

            reward_transform_class=None,
            reward_transform_kwargs=None,
            terminal_transform_class=None,
            terminal_transform_kwargs=None,

            pretraining_env_logging_period=100000,
            pretraining_logging_period=1000,
            do_pretrain_rollouts=False,

            train_bc_on_rl_buffer=False,
            use_automatic_beta_tuning=False,
            beta_epsilon=1e-10,
            normalize_over_batch=True,
            normalize_over_state="advantage",
            Z_K=10,
            clip_score=None,
            validation_qlearning=False,
    ):
        super().__init__()
        self.env = env
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.buffer_policy = buffer_policy
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period

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
        self.bc_loss_type = bc_loss_type
        self.awr_loss_type = awr_loss_type
        self.rl_weight = rl_weight
        self.bc_weight = bc_weight
        self.save_bc_policies = save_bc_policies
        self.eval_policy = MakeDeterministic(self.policy)
        self.compute_bc = compute_bc
        self.alpha = alpha
        self.q_update_period = q_update_period
        self.policy_update_period = policy_update_period
        self.weight_loss = weight_loss

        self.reparam_weight = reparam_weight
        self.awr_weight = awr_weight
        self.klac_weight = klac_weight
        self.klac_K = klac_K
        self.post_pretrain_hyperparams = post_pretrain_hyperparams
        self.post_bc_pretrain_hyperparams = post_bc_pretrain_hyperparams
        self.update_policy = True
        self.pretraining_env_logging_period = pretraining_env_logging_period
        self.pretraining_logging_period = pretraining_logging_period
        self.do_pretrain_rollouts = do_pretrain_rollouts
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
        self.use_klac_update = use_klac_update
        self.clip_score = clip_score

        self.train_bc_on_rl_buffer = train_bc_on_rl_buffer and buffer_policy
        self.validation_qlearning = validation_qlearning

    def get_batch_from_buffer(self, replay_buffer, batch_size):
        batch = replay_buffer.random_batch(batch_size)
        batch = np_to_pytorch_batch(batch)
        return batch

    def run_bc_batch(self, replay_buffer, policy):
        batch = self.get_batch_from_buffer(replay_buffer, self.bc_batch_size)
        o = batch["observations"]
        u = batch["actions"]
        # g = batch["resampled_goals"]
        # og = torch.cat((o, g), dim=1)
        og = o
        # pred_u, *_ = self.policy(og)
        dist = policy(og)
        pred_u, log_pi = dist.rsample_and_logprob()
        stats = dist.get_diagnostics()

        mse = (pred_u - u) ** 2
        mse_loss = mse.mean()

        policy_logpp = dist.log_prob(u, )
        logp_loss = -policy_logpp.mean()

        # T = 0
        if self.bc_loss_type == "mle":
            policy_loss = logp_loss
        elif self.bc_loss_type == "mse":
            policy_loss = mse_loss
        else:
            error

        return policy_loss, logp_loss, mse_loss, stats

    def do_rollouts(self):
        total_ret = 0
        for _ in range(20):
            o = self.env.reset()
            ret = 0
            for _ in range(1000):
                a, _ = self.policy.get_action(o)
                o, r, done, info = self.env.step(a)
                ret += r
                if done:
                    break
            total_ret += ret
        return total_ret

    def pretrain_policy_with_bc(self, policy, train_buffer, test_buffer, steps, label="policy", ):
        logger.remove_tabular_output(
            'progress.csv', relative_to_snapshot_dir=True,
        )
        logger.add_tabular_output(
            'pretrain_%s.csv' % label, relative_to_snapshot_dir=True,
        )
        if self.do_pretrain_rollouts:
            total_ret = self.do_rollouts()
            print("INITIAL RETURN", total_ret/20)

        optimizer = self.optimizers[policy]
        prev_time = time.time()
        for i in range(steps):
            train_policy_loss, train_logp_loss, train_mse_loss, train_stats = self.run_bc_batch(train_buffer, policy)
            train_policy_loss = train_policy_loss * self.bc_weight

            optimizer.zero_grad()
            train_policy_loss.backward()
            optimizer.step()

            test_policy_loss, test_logp_loss, test_mse_loss, test_stats = self.run_bc_batch(test_buffer, policy)
            test_policy_loss = test_policy_loss * self.bc_weight

            if self.do_pretrain_rollouts and i % self.pretraining_env_logging_period == 0:
                total_ret = self.do_rollouts()
                print("Return at step {} : {}".format(i, total_ret/20))

            if i % self.pretraining_logging_period==0:
                stats = {
                "pretrain_bc/batch": i,
                "pretrain_bc/Train Logprob Loss": ptu.get_numpy(train_logp_loss),
                "pretrain_bc/Test Logprob Loss": ptu.get_numpy(test_logp_loss),
                "pretrain_bc/Train MSE": ptu.get_numpy(train_mse_loss),
                "pretrain_bc/Test MSE": ptu.get_numpy(test_mse_loss),
                "pretrain_bc/train_policy_loss": ptu.get_numpy(train_policy_loss),
                "pretrain_bc/test_policy_loss": ptu.get_numpy(test_policy_loss),
                "pretrain_bc/epoch_time":time.time()-prev_time,
                }

                if self.do_pretrain_rollouts:
                    stats["pretrain_bc/avg_return"] = total_ret / 20

                logger.record_dict(stats)
                logger.dump_tabular(with_prefix=True, with_timestamp=False)
                pickle.dump(self.policy, open(logger.get_snapshot_dir() + '/bc_%s.pkl' % label, "wb"))
                prev_time = time.time()

        logger.remove_tabular_output(
            'pretrain_%s.csv' % label, relative_to_snapshot_dir=True,
        )
        logger.add_tabular_output(
            'progress.csv', relative_to_snapshot_dir=True,
        )

        if self.post_bc_pretrain_hyperparams:
            self.set_algorithm_weights(**self.post_bc_pretrain_hyperparams)

    def pretrain_q_with_bc_data(self):
        logger.remove_tabular_output(
            'progress.csv', relative_to_snapshot_dir=True
        )
        logger.add_tabular_output(
            'pretrain_q.csv', relative_to_snapshot_dir=True
        )

        self.update_policy = False
        # first train only the Q function
        for i in range(self.q_num_pretrain1_steps):
            self.eval_statistics = dict()

            train_data = self.replay_buffer.random_batch(self.bc_batch_size)
            train_data = np_to_pytorch_batch(train_data)
            obs = train_data['observations']
            next_obs = train_data['next_observations']
            # goals = train_data['resampled_goals']
            train_data['observations'] = obs # torch.cat((obs, goals), dim=1)
            train_data['next_observations'] = next_obs # torch.cat((next_obs, goals), dim=1)
            self.train_from_torch(train_data)
            if i%self.pretraining_logging_period == 0:
                stats_with_prefix = add_prefix(self.eval_statistics, prefix="trainer/")
                logger.record_dict(stats_with_prefix)
                logger.dump_tabular(with_prefix=True, with_timestamp=False)

        self.update_policy = True
        # then train policy and Q function together
        prev_time = time.time()
        for i in range(self.q_num_pretrain2_steps):
            self.eval_statistics = dict()
            if i % self.pretraining_logging_period == 0:
                self._need_to_update_eval_statistics=True
            train_data = self.replay_buffer.random_batch(self.bc_batch_size)
            train_data = np_to_pytorch_batch(train_data)
            obs = train_data['observations']
            next_obs = train_data['next_observations']
            # goals = train_data['resampled_goals']
            train_data['observations'] = obs # torch.cat((obs, goals), dim=1)
            train_data['next_observations'] = next_obs # torch.cat((next_obs, goals), dim=1)
            self.train_from_torch(train_data)
            if self.do_pretrain_rollouts and i % self.pretraining_env_logging_period == 0:
                total_ret = self.do_rollouts()
                print("Return at step {} : {}".format(i, total_ret/20))

            if i%self.pretraining_logging_period==0:
                if self.do_pretrain_rollouts:
                    self.eval_statistics["pretrain_bc/avg_return"] = total_ret / 20
                self.eval_statistics["batch"] = i
                self.eval_statistics["epoch_time"] = time.time()-prev_time
                stats_with_prefix = add_prefix(self.eval_statistics, prefix="trainer/")
                logger.record_dict(stats_with_prefix)
                logger.dump_tabular(with_prefix=True, with_timestamp=False)
                prev_time = time.time()

        logger.remove_tabular_output(
            'pretrain_q.csv',
            relative_to_snapshot_dir=True,
        )
        logger.add_tabular_output(
            'progress.csv',
            relative_to_snapshot_dir=True,
        )

        self._need_to_update_eval_statistics = True
        self.eval_statistics = dict()

        if self.post_pretrain_hyperparams:
            self.set_algorithm_weights(**self.post_pretrain_hyperparams)

    def set_algorithm_weights(
        self,
        # bc_weight,
        # rl_weight,
        # use_awr_update,
        # use_reparam_update,
        # reparam_weight,
        # awr_weight,
        **kwargs
    ):
        for key in kwargs:
            self.__dict__[key] = kwargs[key]
        # self.bc_weight = bc_weight
        # self.rl_weight = rl_weight
        # self.use_awr_update = use_awr_update
        # self.use_reparam_update = use_reparam_update
        # self.awr_weight = awr_weight

    def test_from_torch(self, batch):
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
        new_obs_actions, policy_mean, policy_log_std, log_pi, entropy, policy_std, mean_action_log_prob, pretanh_value, dist = self.policy(
            obs, reparameterize=True, return_log_prob=True,
        )

        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = self.alpha

        q1_pred = self.qf1(obs, actions)
        q2_pred = self.qf2(obs, actions)
        # Make sure policy accounts for squashing functions like tanh correctly!
        new_next_actions, _, _, new_log_pi, *_ = self.policy(
            next_obs, reparameterize=True, return_log_prob=True,
        )
        target_q_values = torch.min(
            self.target_qf1(next_obs, new_next_actions),
            self.target_qf2(next_obs, new_next_actions),
        ) - alpha * new_log_pi

        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
        qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
        qf2_loss = self.qf_criterion(q2_pred, q_target.detach())

        qf1_new_actions = self.qf1(obs, new_obs_actions)
        qf2_new_actions = self.qf2(obs, new_obs_actions)
        q_new_actions = torch.min(
            qf1_new_actions,
            qf2_new_actions,
        )

        policy_loss = (log_pi - q_new_actions).mean()

        self.eval_statistics['validation/QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
        self.eval_statistics['validation/QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
        self.eval_statistics['validation/Policy Loss'] = np.mean(ptu.get_numpy(
            policy_loss
        ))
        self.eval_statistics.update(create_stats_ordered_dict(
            'validation/Q1 Predictions',
            ptu.get_numpy(q1_pred),
        ))
        self.eval_statistics.update(create_stats_ordered_dict(
            'validation/Q2 Predictions',
            ptu.get_numpy(q2_pred),
        ))
        self.eval_statistics.update(create_stats_ordered_dict(
            'validation/Q Targets',
            ptu.get_numpy(q_target),
        ))
        self.eval_statistics.update(create_stats_ordered_dict(
            'validation/Log Pis',
            ptu.get_numpy(log_pi),
        ))
        self.eval_statistics.update(create_stats_ordered_dict(
            'validation/Policy mu',
            ptu.get_numpy(policy_mean),
        ))
        self.eval_statistics.update(create_stats_ordered_dict(
            'validation/Policy log std',
            ptu.get_numpy(policy_log_std),
        ))

    def train_from_torch(self, batch, train=True):
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
        policy_mean = dist.get_mle()

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
            v1_pi = self.qf1(obs, policy_mean)
            v2_pi = self.qf2(obs, policy_mean)
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
        else:
            u = actions
            if self.awr_min_q:
                q_adv = torch.min(q1_pred, q2_pred)
            else:
                q_adv = q1_pred

        if self.awr_loss_type == "mse":
            policy_logpp = -(policy_mean - actions) ** 2
        else:
            policy_logpp = dist.log_prob(u)
            policy_logpp = policy_logpp.sum(dim=1, keepdim=True)

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
        elif self.normalize_over_state == "Z":
            *_, buffer_dist = self.buffer_policy(
                obs, reparameterize=True, return_log_prob=True,
            )
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
            elif self.normalize_over_batch == False:
                weights = score
            else:
                error

        policy_loss = alpha * log_pi.mean()


        if self.use_klac_update:
            *_, buffer_dist = self.buffer_policy(
                obs, reparameterize=True, return_log_prob=True,
            )
            K = self.klac_K
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
            q_max = q_b.max(dim=1)[0][:, None]
            q_normalized = q_b - q_max
            q_weights_b = F.softmax(q_normalized / beta, dim=1).flatten() * K
            # klac_loss = (-log_pi * q_weights_b.detach() / p_buffer.detach()).mean()
            klac_loss = (-log_pi * q_weights_b.detach()).mean()
            policy_loss = policy_loss + self.klac_weight * klac_loss

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

        if self.train_bc_on_rl_buffer:
            buffer_policy_loss, buffer_train_logp_loss, buffer_train_mse_loss, _ = self.run_bc_batch(self.replay_buffer.train_replay_buffer, self.buffer_policy)
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
                test_policy_loss, test_logp_loss, test_mse_loss, _ = self.run_bc_batch(self.replay_buffer.validation_replay_buffer,
                                                                                       self.buffer_policy)
                buffer_dist = self.buffer_policy(obs)

                kldiv = torch.distributions.kl.kl_divergence(dist, buffer_dist)

                self.eval_statistics.update({
                    "buffer_policy/Train Logprob Loss": ptu.get_numpy(buffer_train_logp_loss),
                    "buffer_policy/Test Logprob Loss": ptu.get_numpy(test_logp_loss),
                    "buffer_policy/Train MSE": ptu.get_numpy(buffer_train_mse_loss),
                    "buffer_policy/Test MSE": ptu.get_numpy(test_mse_loss),
                    "buffer_policy/train_policy_loss": ptu.get_numpy(buffer_policy_loss),
                    "buffer_policy/test_policy_loss": ptu.get_numpy(test_policy_loss),
                    "buffer_policy/kl_div":ptu.get_numpy(kldiv.mean()),
                })
            if self.use_automatic_beta_tuning:
                self.eval_statistics.update({
                    "adaptive_beta/beta":ptu.get_numpy(beta.mean()),
                    "adaptive_beta/beta loss": ptu.get_numpy(beta_loss.mean()),
                })
            if self.use_klac_update:
                self.eval_statistics.update({
                    "klac/loss": ptu.get_numpy(klac_loss.mean()),
                })
                self.eval_statistics.update(create_stats_ordered_dict(
                    'klac/weights',
                    ptu.get_numpy(q_weights_b),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'klac/p_buffer',
                    ptu.get_numpy(p_buffer),
                ))

            # if self.validation_qlearning:
            #     train_data = self.replay_buffer.validation_replay_buffer.random_batch(self.bc_batch_size)
            #     train_data = np_to_pytorch_batch(train_data)
            #     obs = train_data['observations']
            #     next_obs = train_data['next_observations']
            #     # goals = train_data['resampled_goals']
            #     train_data['observations'] = obs # torch.cat((obs, goals), dim=1)
            #     train_data['next_observations'] = next_obs # torch.cat((next_obs, goals), dim=1)
            #     self.test_from_torch(train_data)

        self._n_train_steps_total += 1

    def get_diagnostics(self):
        stats = super().get_diagnostics()
        stats.update(self.eval_statistics)
        return stats

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        nets = [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
        ]
        if self.buffer_policy:
            nets.append(self.buffer_policy)
        return nets

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.qf1,
            target_qf2=self.qf2,
            buffer_policy=self.buffer_policy,
        )
