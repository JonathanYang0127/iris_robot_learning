import torch
import numpy as np
from torch import nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

from railrl.torch.bnlstm import BNLSTMCell
from railrl.torch.core import PyTorchModule
from railrl.torch.pytorch_util import FloatTensor, from_numpy, get_numpy, \
    fanin_init


class MemoryPolicy(PyTorchModule):
    def __init__(
            self,
            obs_dim,
            action_dim,
            memory_dim,
            fc1_size,
            fc2_size,
    ):
        self.save_init_params(locals())
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.memory_dim = memory_dim
        self.fc1_size = fc1_size
        self.fc2_size = fc2_size

        self.fc1 = nn.Linear(obs_dim + memory_dim, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.last_fc = nn.Linear(fc2_size, action_dim)
        # self.lstm_cell = nn.LSTMCell(self.obs_dim, self.memory_dim // 2)
        self.lstm_cell = BNLSTMCell(self.obs_dim, self.memory_dim // 2)

    def action_parameters(self):
        for fc in [self.fc1, self.fc2, self.last_fc]:
            for param in fc.parameters():
                yield param

    def write_parameters(self):
        return self.lstm_cell.parameters()

    def forward(self, obs, initial_memory):
        """
        :param obs: torch Variable, [batch_size, sequence length, obs dim]
        :param initial_memory: torch Variable, [batch_size, memory dim]
        :return: (actions, writes) tuple
            actions: [batch_size, sequence length, action dim]
            writes: [batch_size, sequence length, memory dim]
        """
        assert len(obs.size()) == 3
        assert len(initial_memory.size()) == 2
        batch_size, subsequence_length = obs.size()[:2]

        """
        Create the new writes.
        """
        hx, cx = torch.split(initial_memory, self.memory_dim // 2, dim=1)
        new_hxs = Variable(
            FloatTensor(batch_size, subsequence_length, self.memory_dim // 2)
        )
        new_cxs = Variable(
            FloatTensor(batch_size, subsequence_length, self.memory_dim // 2)
        )
        for i in range(subsequence_length):
            hx, cx = self.lstm_cell(obs[:, i, :], (hx, cx))
            new_hxs[:, i, :] = hx
            new_cxs[:, i, :] = cx
        subtraj_writes = torch.cat((new_hxs, new_cxs), dim=2)

        # The reason that using a LSTM doesn't work is that this gives you only
        # the FINAL hx and cx, not all of them :(
        # _, (new_hxs, new_cxs) = self.lstm(obs, (hx, cx))
        # subtraj_writes = torch.cat((new_hxs, new_cxs), dim=2)
        # subtraj_writes = subtraj_writes.permute(1, 0, 2)

        """
        Create the new subtrajectory memories with the initial memories and the
        new writes.
        """
        expanded_init_memory = initial_memory.unsqueeze(1)
        if subsequence_length > 1:
            memories = torch.cat(
                (
                    expanded_init_memory,
                    subtraj_writes[:, :-1, :],
                ),
                dim=1,
            )
        else:
            memories = expanded_init_memory

        """
        Use new memories to create env actions.
        """
        all_subtraj_inputs = torch.cat([obs, memories], dim=2)
        # noinspection PyArgumentList
        subtraj_actions = Variable(
            FloatTensor(batch_size, subsequence_length, self.action_dim)
        )
        for i in range(subsequence_length):
            all_inputs = all_subtraj_inputs[:, i, :]
            h1 = F.tanh(self.fc1(all_inputs))
            h2 = F.tanh(self.fc2(h1))
            action = F.tanh(self.last_fc(h2))
            subtraj_actions[:, i, :] = action

        return subtraj_actions, subtraj_writes

    def get_action(self, augmented_obs):
        """
        :param augmented_obs: (obs, memories) tuple
            obs: np.ndarray, [obs_dim]
            memories: nd.ndarray, [memory_dim]
        :return: (actions, writes) tuple
            actions: np.ndarray, [action_dim]
            writes: np.ndarray, [writes_dim]
        """
        obs, memory = augmented_obs
        obs = np.expand_dims(obs, axis=0)
        memory = np.expand_dims(memory, axis=0)
        obs = Variable(from_numpy(obs).float(), requires_grad=False)
        memory = Variable(from_numpy(memory).float(), requires_grad=False)
        action, write = self.get_flat_output(obs, memory)
        return (
                   np.squeeze(get_numpy(action), axis=0),
                   np.squeeze(get_numpy(write), axis=0),
               ), {}

    def get_flat_output(self, obs, initial_memories):
        """
        Each batch element is processed independently. So, there's no recurrency
        used.

        :param obs: torch Variable, [batch_size X obs_dim]
        :param initial_memories: torch Variable, [batch_size X memory_dim]
        :return: (actions, writes) Tuple
            actions: torch Variable, [batch_size X action_dim]
            writes: torch Variable, [batch_size X writes_dim]
        """
        obs = obs.unsqueeze(1)
        actions, writes = self.__call__(obs, initial_memories)
        return torch.squeeze(actions, dim=1), torch.squeeze(writes, dim=1)

    def reset(self):
        pass

    def log_diagnostics(self):
        pass


class FeedForwardPolicy(PyTorchModule):
    def __init__(
            self,
            obs_dim,
            action_dim,
            fc1_size,
            fc2_size,
            init_w=1e-3,
    ):
        self.save_init_params(locals())
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.fc1_size = fc1_size
        self.fc2_size = fc2_size

        self.fc1 = nn.Linear(obs_dim, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.last_fc = nn.Linear(fc2_size, action_dim)

        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc1.bias.data *= 0
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc2.bias.data *= 0
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, obs):
        h = F.relu(self.fc1(obs))
        h = F.relu(self.fc2(h))
        return F.tanh(self.last_fc(h))

    def get_action(self, obs):
        obs = np.expand_dims(obs, axis=0)
        obs = Variable(from_numpy(obs).float(), requires_grad=False)
        action = self.__call__(obs)
        action = action.squeeze(0)
        if self.last_fc.weight.is_cuda:
            return action.data.cpu().numpy(), {}
        else:
            return action.data.numpy(), {}

    def reset(self):
        pass

    def log_diagnostics(self, paths):
        pass