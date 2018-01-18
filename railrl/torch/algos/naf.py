from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

from railrl.samplers.util import split_paths_to_dict
from railrl.torch import pytorch_util as ptu
from railrl.torch.algos.util import np_to_pytorch_batch
from railrl.torch.core import PyTorchModule
from railrl.misc.eval_util import (
    get_difference_statistics, get_average_returns,
    create_stats_ordered_dict)
from railrl.torch.algos.torch_rl_algorithm import TorchRLAlgorithm
from railrl.core import logger


# noinspection PyCallingNonCallable
class NAF(TorchRLAlgorithm):
    def __init__(
            self,
            env,
            policy,
            exploration_policy=None,
            policy_learning_rate=1e-3,
            target_hard_update_period=1000,
            tau=0.001,
            use_soft_update=False,
            **kwargs
    ):
        if exploration_policy is None:
            exploration_policy = policy
        super().__init__(
            env,
            exploration_policy,
            **kwargs
        )
        self.policy = policy
        self.policy_learning_rate = policy_learning_rate
        self.target_policy = self.policy.copy()
        self.target_hard_update_period = target_hard_update_period
        self.tau = tau
        self.use_soft_update = use_soft_update

        self.policy_criterion = nn.MSELoss()
        self.policy_optimizer = optim.Adam(
            self.policy.parameters(),
            lr=self.policy_learning_rate,
        )

    def _do_training(self):
        batch = self.get_batch()

        """
        Optimize Critic.
        """
        train_dict = self.get_train_dict(batch)
        policy_loss = train_dict['Policy Loss']

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        """
        Update Target Networks
        """
        if self.use_soft_update:
            ptu.soft_update_from_to(self.policy, self.target_policy, self.tau)
        else:
            if self._n_train_steps_total% self.target_hard_update_period == 0:
                ptu.copy_model_params_from_to(self.policy, self.target_policy)

    def get_train_dict(self, batch):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        _, _, v_pred = self.target_policy(next_obs, None)
        y_target = rewards + (1. - terminals) * self.discount * v_pred
        # noinspection PyUnresolvedReferences
        y_target = y_target.detach()
        mu, y_pred, v = self.policy(obs, actions)
        policy_loss = self.policy_criterion(y_pred, y_target)

        return OrderedDict([
            ('Policy v', v),
            ('Policy mu', mu),
            ('Policy Loss', policy_loss),
            ('Y targets', y_target),
            ('Y predictions', y_pred),
        ])

    def evaluate(self, epoch):
        """
        Perform evaluation for this algorithm.

        :param epoch: The epoch number.
        """
        logger.log("Collecting samples for evaluation")
        train_batch = self.get_batch(training=True)
        validation_batch = self.get_batch(training=False)
        test_paths = self.eval_sampler.obtain_samples()

        statistics = OrderedDict()
        statistics.update(
            self._statistics_from_paths(self._exploration_paths, "Exploration")
        )
        statistics.update(self._statistics_from_paths(test_paths, "Test"))
        statistics.update(self._statistics_from_batch(train_batch, "Train"))
        statistics.update(
            self._statistics_from_batch(validation_batch, "Validation")
        )
        statistics.update(
            get_difference_statistics(statistics, ['Policy Loss Mean'])
        )
        statistics['AverageReturn'] = get_average_returns(test_paths)
        statistics['Epoch'] = epoch

        for key, value in statistics.items():
            logger.record_tabular(key, value)

        self.env.log_diagnostics(test_paths)

    def _statistics_from_paths(self, paths, stat_prefix):
        np_batch = split_paths_to_dict(paths)
        batch = np_to_pytorch_batch(np_batch)
        statistics = self._statistics_from_batch(batch, stat_prefix)
        statistics.update(create_stats_ordered_dict(
            'Num Paths', len(paths), stat_prefix=stat_prefix
        ))
        return statistics

    def _statistics_from_batch(self, batch, stat_prefix):
        statistics = get_statistics_from_pytorch_dict(
            self.get_train_dict(batch),
            ['Policy Loss'],
            ['Policy v', 'Policy mu', 'Y targets', 'Y predictions'],
            stat_prefix
        )
        statistics.update(create_stats_ordered_dict(
            "{} Env Actions".format(stat_prefix),
            ptu.get_numpy(batch['actions'])
        ))

        return statistics

    def get_epoch_snapshot(self, epoch):
        return dict(
            epoch=epoch,
            env=self.training_env,
            policy=self.policy,
            replay_buffer=self.replay_buffer,
            algorithm=self,
        )

    @property
    def networks(self):
        return [
            self.policy,
            self.target_policy,
        ]


class NafPolicy(PyTorchModule):
    def __init__(
            self,
            obs_dim,
            action_dim,
            hidden_size,
            use_batchnorm=False,
            b_init_value=0.01,
            hidden_init=ptu.fanin_init,
            use_exp_for_diagonal_not_square=True,
    ):
        self.save_init_params(locals())
        super(NafPolicy, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.use_batchnorm = use_batchnorm
        self.use_exp_for_diagonal_not_square = use_exp_for_diagonal_not_square

        if use_batchnorm:
            self.bn_state = nn.BatchNorm1d(obs_dim)
            self.bn_state.weight.data.fill_(1)
            self.bn_state.bias.data.fill_(0)

        self.linear1 = nn.Linear(obs_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)
        self.mu = nn.Linear(hidden_size, action_dim)
        self.L = nn.Linear(hidden_size, action_dim ** 2)

        self.tril_mask = ptu.Variable(
            torch.tril(
                torch.ones(action_dim, action_dim),
                -1
            ).unsqueeze(0)
        )
        self.diag_mask = ptu.Variable(torch.diag(
            torch.diag(
                torch.ones(action_dim, action_dim)
            )
        ).unsqueeze(0))

        hidden_init(self.linear1.weight)
        self.linear1.bias.data.fill_(b_init_value)
        hidden_init(self.linear2.weight)
        self.linear2.bias.data.fill_(b_init_value)
        hidden_init(self.V.weight)
        self.V.bias.data.fill_(b_init_value)
        hidden_init(self.L.weight)
        self.L.bias.data.fill_(b_init_value)
        hidden_init(self.mu.weight)
        self.mu.bias.data.fill_(b_init_value)

    def forward(self, state, action, return_P=False):
        if self.use_batchnorm:
            state = self.bn_state(state)
        x = state
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))

        V = self.V(x)
        mu = torch.tanh(self.mu(x))

        Q = None
        P = None
        if action is not None or return_P:
            num_outputs = mu.size(1)
            raw_L = self.L(x).view(-1, num_outputs, num_outputs)
            L = raw_L * self.tril_mask.expand_as(raw_L)
            if self.use_exp_for_diagonal_not_square:
                L += torch.exp(raw_L) * self.diag_mask.expand_as(raw_L)
            else:
                L += torch.pow(raw_L, 2) * self.diag_mask.expand_as(raw_L)
            P = torch.bmm(L, L.transpose(2, 1))

            if action is not None:
                u_mu = (action - mu).unsqueeze(2)
                A = - 0.5 * torch.bmm(
                    torch.bmm(u_mu.transpose(2, 1), P), u_mu
                ).squeeze(2)

                Q = A + V

        if return_P:
            return mu, Q, V, P
        return mu, Q, V

    def get_action(self, obs):
        obs = np.expand_dims(obs, axis=0)
        obs = Variable(ptu.from_numpy(obs).float(), requires_grad=False)
        action, _, _ = self.__call__(obs, None)
        action = action.squeeze(0)
        return ptu.get_numpy(action), {}

    def get_action_and_P_matrix(self, obs):
        obs = np.expand_dims(obs, axis=0)
        obs = Variable(ptu.from_numpy(obs).float(), requires_grad=False)
        action, _, _, P = self.__call__(obs, None, return_P=True)
        action = action.squeeze(0)
        P = P.squeeze(0)
        return ptu.get_numpy(action), ptu.get_numpy(P)


def get_statistics_from_pytorch_dict(
        pytorch_dict,
        mean_stat_names,
        full_stat_names,
        stat_prefix,
):
    """
    :param pytorch_dict: Dictionary, from string to pytorch Tensor
    :param mean_stat_names: List of strings. Add the mean of these
    Tensors to the output
    :param full_stat_names: List of strings. Add all statistics of these
    Tensors to the output
    :param stat_prefix: Prefix to all statistics in outputted dict.
    :return: OrderedDict of statistics
    """
    statistics = OrderedDict()
    for name in mean_stat_names:
        tensor = pytorch_dict[name]
        statistics_name = "{} {} Mean".format(stat_prefix, name)
        statistics[statistics_name] = np.mean(ptu.get_numpy(tensor))

    for name in full_stat_names:
        tensor = pytorch_dict[name]
        data = ptu.get_numpy(tensor)
        statistics.update(create_stats_ordered_dict(
            '{} {}'.format(stat_prefix, name),
            data,
        ))
    return statistics


def get_difference_statistics(
        statistics,
        stat_names,
        include_validation_train_gap=True,
        include_test_validation_gap=True,
):
    assert include_validation_train_gap or include_test_validation_gap
    difference_pairs = []
    if include_validation_train_gap:
        difference_pairs.append(('Validation', 'Train'))
    if include_test_validation_gap:
        difference_pairs.append(('Test', 'Validation'))
    differences = OrderedDict()
    for prefix_1, prefix_2 in difference_pairs:
        for stat_name in stat_names:
            diff_name = "{0}: {1} - {2}".format(
                stat_name,
                prefix_1,
                prefix_2,
            )
            differences[diff_name] = (
                    statistics["{0} {1}".format(prefix_1, stat_name)]
                    - statistics["{0} {1}".format(prefix_2, stat_name)]
            )
    return differences


