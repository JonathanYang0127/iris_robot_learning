import copy
import numpy as np

import torch
from torch import nn as nn
import torch.nn.functional as F

import rlkit.torch.pytorch_util as ptu
from rlkit.core.util import Wrapper
from rlkit.policies.base import ExplorationPolicy, Policy
from rlkit.torch.core import np_ify
from rlkit.torch.distributions import TanhNormal
from rlkit.torch.networks import Mlp
from rlkit.torch.networks.stochastic.distribution_generator import \
    DistributionGenerator
from rlkit.torch.sac.policies import MakeDeterministic, TorchStochasticPolicy


def _product_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of product of gaussians
    '''
    sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)
    sigma_squared = 1. / torch.sum(torch.reciprocal(sigmas_squared), dim=0)
    mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=0)
    return mu, sigma_squared


def _mean_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of mean of gaussians
    '''
    mu = torch.mean(mus, dim=0)
    sigma_squared = torch.mean(sigmas_squared, dim=0)
    return mu, sigma_squared


def _natural_to_canonical(n1, n2):
    ''' convert from natural to canonical gaussian parameters '''
    mu = -0.5 * n1 / n2
    sigma_squared = -0.5 * 1 / n2
    return mu, sigma_squared


def _canonical_to_natural(mu, sigma_squared):
    ''' convert from canonical to natural gaussian parameters '''
    n1 = mu / sigma_squared
    n2 = -0.5 * 1 / sigma_squared
    return n1, n2


class PEARLAgent(nn.Module):

    def __init__(self,
                 latent_dim,
                 context_encoder,
                 policy,
                 reward_predictor,
                 **kwargs
                 ):
        super().__init__()
        self.latent_dim = latent_dim

        self.context_encoder = context_encoder
        self.policy = policy
        self.reward_predictor = reward_predictor
        self.deterministic_policy = MakeDeterministic(self.policy)

        self.recurrent = kwargs['recurrent']
        self.use_ib = kwargs['use_information_bottleneck']
        self.sparse_rewards = kwargs['sparse_rewards']
        self.use_next_obs_in_context = kwargs['use_next_obs_in_context']

        # initialize buffers for z dist and z
        # use buffers so latent context can be saved along with model weights
        self.register_buffer('z', torch.zeros(1, latent_dim))
        self.register_buffer('z_means', torch.zeros(1, latent_dim))
        self.register_buffer('z_vars', torch.zeros(1, latent_dim))

        self.z_means = None
        self.z_vars = None
        self.context = None
        self.z = None

        # rp = reward predictor
        self.z_means_rp = None
        self.z_vars_rp = None
        self.z_rp = None
        self.context_encoder_rp = context_encoder
        self._in_unsupervised_phase = False

        self.clear_z()

    @property
    def use_context_encoder_snapshot_for_reward_pred(self):
        return self._in_unsupervised_phase

    @use_context_encoder_snapshot_for_reward_pred.setter
    def use_context_encoder_snapshot_for_reward_pred(self, value):
        if value and not self.use_context_encoder_snapshot_for_reward_pred:
            # copy context encoder on switch
            self.context_encoder_rp = copy.deepcopy(self.context_encoder)
        self._in_unsupervised_phase = value

    def clear_z(self, num_tasks=1):
        '''
        reset q(z|c) to the prior
        sample a new z from the prior
        '''
        # reset distribution over z to the prior
        mu = ptu.zeros(num_tasks, self.latent_dim)
        if self.use_ib:
            var = ptu.ones(num_tasks, self.latent_dim)
        else:
            var = ptu.zeros(num_tasks, self.latent_dim)
        self.z_means = mu
        self.z_vars = var
        self.z_means_rp = mu
        self.z_vars_rp = var
        # sample a new z from the prior
        self.sample_z()
        # reset the context collected so far
        self.context = None
        # reset any hidden state in the encoder network (relevant for RNN)
        self.context_encoder.reset(num_tasks)

    def detach_z(self):
        ''' disable backprop through z '''
        self.z = self.z.detach()
        if self.recurrent:
            self.context_encoder.hidden = self.context_encoder.hidden.detach()

        self.z_rp = self.z_rp.detach()
        if self.recurrent:
            self.context_encoder_rp.hidden = self.context_encoder_rp.hidden.detach()

    def update_context(self, inputs):
        ''' append single transition to the current context '''
        o, a, r, no, d, info = inputs
        if self.sparse_rewards:
            r = info['sparse_reward']
        o = ptu.from_numpy(o[None, None, ...])
        a = ptu.from_numpy(a[None, None, ...])
        r = ptu.from_numpy(np.array([r])[None, None, ...])
        no = ptu.from_numpy(no[None, None, ...])

        if self.use_next_obs_in_context:
            data = torch.cat([o, a, r, no], dim=2)
        else:
            data = torch.cat([o, a, r], dim=2)
        if self.context is None:
            self.context = data
        else:
            self.context = torch.cat([self.context, data], dim=1)

    def compute_kl_div(self):
        ''' compute KL( q(z|c) || r(z) ) '''
        prior = torch.distributions.Normal(ptu.zeros(self.latent_dim), ptu.ones(self.latent_dim))
        posteriors = [torch.distributions.Normal(mu, torch.sqrt(var)) for mu, var in zip(torch.unbind(self.z_means), torch.unbind(self.z_vars))]
        kl_divs = [torch.distributions.kl.kl_divergence(post, prior) for post in posteriors]
        kl_div_sum = torch.sum(torch.stack(kl_divs))
        return kl_div_sum

    def infer_posterior(self, context):
        ''' compute q(z|c) as a function of input context and sample new z from it'''
        try:
            params = self.context_encoder(context)
            params = params.view(context.size(0), -1, self.context_encoder.output_size)
        except TypeError:
            import ipdb; ipdb.set_trace()
        # with probabilistic z, predict mean and variance of q(z | c)
        if self.use_ib:
            mu = params[..., :self.latent_dim]
            sigma_squared = F.softplus(params[..., self.latent_dim:])
            z_params = [_product_of_gaussians(m, s) for m, s in zip(torch.unbind(mu), torch.unbind(sigma_squared))]
            self.z_means = torch.stack([p[0] for p in z_params])
            self.z_vars = torch.stack([p[1] for p in z_params])
        # sum rather than product of gaussians structure
        else:
            self.z_means = torch.mean(params, dim=1)
        if self.use_context_encoder_snapshot_for_reward_pred:
            params_rp = self.context_encoder_rp(context)
            params_rp = params_rp.view(context.size(0), -1, self.context_encoder_rp.output_size)
            if self.use_ib:
                mu_rp = params_rp[..., :self.latent_dim]
                sigma_squared_rp = F.softplus(params_rp[..., self.latent_dim:])
                z_params_rp = [_product_of_gaussians(m, s) for m, s in zip(torch.unbind(mu_rp), torch.unbind(sigma_squared_rp))]
                self.z_means_rp = torch.stack([p[0] for p in z_params_rp])
                self.z_vars_rp = torch.stack([p[1] for p in z_params_rp])
            # sum rather than product of gaussians structure
            else:
                self.z_means_rp = torch.mean(params_rp, dim=1)
        else:
            self.z_means_rp = self.z_means
            self.z_vars_rp = self.z_vars
        self.sample_z()

    def sample_z(self):
        if self.use_ib:
            posteriors = [torch.distributions.Normal(m, torch.sqrt(s)) for m, s in zip(torch.unbind(self.z_means), torch.unbind(self.z_vars))]
            z = [d.rsample() for d in posteriors]
            self.z = torch.stack(z)
        else:
            self.z = self.z_means
        if self.use_context_encoder_snapshot_for_reward_pred:
            if self.use_ib:
                posteriors_rp = [torch.distributions.Normal(m, torch.sqrt(s)) for m, s in zip(torch.unbind(self.z_means_rp), torch.unbind(self.z_vars_rp))]
                z_rp = [d.rsample() for d in posteriors_rp]
                self.z_rp = torch.stack(z_rp)
            else:
                self.z_rp = self.z_means_rp
        else:
            self.z_rp = self.z

    def get_action(self, obs, deterministic=False):
        ''' sample action from the policy, conditioned on the task embedding '''
        z = self.z
        obs = ptu.from_numpy(obs[None])
        in_ = torch.cat([obs, z], dim=1)[0]
        if deterministic:
            return self.deterministic_policy.get_action(in_)
        else:
            return self.policy.get_action(in_)

    def set_num_steps_total(self, n):
        self.policy.set_num_steps_total(n)

    def forward(self, obs, context, return_task_z=False):
        ''' given context, get statistics under the current policy of a set of observations '''
        self.infer_posterior(context)
        self.sample_z()

        task_z = self.z

        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        task_z = [z.repeat(b, 1) for z in task_z]
        task_z = torch.cat(task_z, dim=0)

        # run policy, get log probs and new actions
        in_ = torch.cat([obs, task_z.detach()], dim=1)
        action_distribution = self.policy(in_)
        # policy_outputs = self.policy(in_, reparameterize=True, return_log_prob=True)
        if return_task_z:
            return action_distribution, task_z
        else:
            return action_distribution

        # return policy_outputs, task_z

    def infer_reward(self, obs, action):
        z = self.z_rp
        obs = ptu.from_numpy(obs[None])
        action = ptu.from_numpy(action[None])
        # in_ = torch.cat([obs, action, z], dim=self.z)
        reward = self.reward_predictor(obs, action, z)
        return ptu.get_numpy(reward)[0]

    def log_diagnostics(self, eval_statistics):
        '''
        adds logging data about encodings to eval_statistics
        '''
        z_mean = np.mean(np.abs(ptu.get_numpy(self.z_means[0])))
        z_sig = np.mean(ptu.get_numpy(self.z_vars[0]))
        eval_statistics['Z mean eval'] = z_mean
        eval_statistics['Z variance eval'] = z_sig

        z_mean_rp = np.mean(np.abs(ptu.get_numpy(self.z_means_rp[0])))
        z_sig_rp = np.mean(ptu.get_numpy(self.z_vars_rp[0]))
        eval_statistics['Z rew-pred mean eval'] = z_mean_rp
        eval_statistics['Z rew-pred variance eval'] = z_sig_rp

    @property
    def networks(self):
        if self.context_encoder is self.context_encoder_rp:
            return [self.context_encoder, self.policy]
        else:
            return [self.context_encoder, self.context_encoder_rp, self.policy]


class MakePEARLAgentDeterministic(Wrapper, Policy):
    def __init__(self, stochastic_policy):
        super().__init__(stochastic_policy)
        self.stochastic_policy = stochastic_policy

    def get_action(self, observation):
        return self.stochastic_policy.get_action(observation,
                                                 deterministic=True)

    def get_actions(self, observations):
        return self.stochastic_policy.get_actions(observations,
                                                  deterministic=True)
