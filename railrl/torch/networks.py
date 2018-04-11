"""
General networks for pytorch.

Algorithm-specific networks should go else-where.
"""
import torch
from torch import nn as nn
from torch.nn import functional as F

from railrl.policies.base import Policy
from railrl.pythonplusplus import identity
from railrl.torch import pytorch_util as ptu
from railrl.torch.core import PyTorchModule
from railrl.torch.data_management.normalizer import TorchFixedNormalizer
from railrl.torch.modules import SelfOuterProductLinear, LayerNorm

from functools import reduce
import pdb

class CNN(PyTorchModule):
    def __init__(self,
                input_size,
                in_channel,
                output_size,
                kernel_sizes,
                n_channels,
                strides,
                pool_sizes,
                paddings,
        ):
        self.save_init_params(locals())
        super().__init__()

        # assume square input. input_size is width or height
        self.input_size = input_size
        self.in_channel = in_channel

        self.layers = []
        output_dim = input_size
        for out_channel, kernel_size, stride, pool, padding in \
            zip(n_channels, kernel_sizes, strides, pool_sizes, paddings):
            conv_layer = nn.Sequential(
                            nn.Conv2d(
                                in_channels=in_channel,
                                out_channels=out_channel,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                            ),
                            nn.ReLU(),
                            nn.MaxPool2d(pool),
                        )
            in_channel = out_channel
            self.layers.append(conv_layer)

            # am very skeptical of this for strides/pools/paddings that aren't one
            output_dim = output_dim - kernel_size + 1
            output_dim //= stride
            output_dim += 2*padding
            output_dim //= pool

        self.out = nn.Linear(int(output_dim**2) * out_channel, output_size)
        self.apply(CNN.init_weights)

    @staticmethod
    def init_weights(m):
        if type(m) == nn.Linear:
            m.weight.data.uniform_(-1e-3, 1e-3)
            m.bias.data.fill_(0)
        if type(m) == nn.Conv2d:
            m.weight.data.normal_(0.0, 0.02)

    def forward(self, input):
        # need to reshape from batch of flattened images into (channsls, w, h)
        h = input.view(input.shape[0], self.in_channel, self.input_size, self.input_size)
        for layer in self.layers:
            h = layer(h)
        h = h.view(h.size(0), -1)
        output = self.out(h)
        return output


class Mlp(PyTorchModule):
    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            hidden_init=ptu.fanin_init,
            b_init_value=0.,
            layer_norm=False,
            layer_norm_kwargs=None,
    ):
        self.save_init_params(locals())
        super().__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.fcs = []
        self.layer_norms = []
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

            if self.layer_norm:
                ln = LayerNorm(next_size)
                self.__setattr__("layer_norm{}".format(i), ln)
                self.layer_norms.append(ln)

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.fill_(0)

    def forward(self, input, return_preactivations=False):
        h = input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output


class FlattenMlp(Mlp):
    """
    Flatten inputs along dimension 1 and then pass through MLP.
    """

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=1)
        return super().forward(flat_inputs, **kwargs)


class MlpQf(FlattenMlp):
    def __init__(
            self,
            *args,
            obs_normalizer: TorchFixedNormalizer = None,
            action_normalizer: TorchFixedNormalizer = None,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(*args, **kwargs)
        self.obs_normalizer = obs_normalizer
        self.action_normalizer = action_normalizer

    def forward(self, obs, actions, **kwargs):
        if self.obs_normalizer:
            obs = self.obs_normalizer.normalize(obs)
        if self.action_normalizer:
            actions = self.action_normalizer.normalize(actions)
        return super().forward(obs, actions, **kwargs)


class MlpPolicy(Mlp, Policy):
    """
    A simpler interface for creating policies.
    """

    def __init__(
            self,
            *args,
            obs_normalizer: TorchFixedNormalizer = None,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(*args, **kwargs)
        self.obs_normalizer = obs_normalizer

    def forward(self, obs, **kwargs):
        if self.obs_normalizer:
            obs = self.obs_normalizer.normalize(obs)
        return super().forward(obs, **kwargs)

    def get_action(self, obs_np):
        actions = self.get_actions(obs_np[None])
        return actions[0, :], {}

    def get_actions(self, obs):
        return self.eval_np(obs)


class TanhMlpPolicy(MlpPolicy):
    """
    A helper class since most policies have a tanh output activation.
    """
    def __init__(self, *args, **kwargs):
        self.save_init_params(locals())
        super().__init__(*args, output_activation=torch.tanh, **kwargs)


class FeedForwardQFunction(PyTorchModule):
    def __init__(
            self,
            obs_dim,
            action_dim,
            observation_hidden_size,
            embedded_hidden_size,
            init_w=3e-3,
            output_activation=identity,
            hidden_init=ptu.fanin_init,
            batchnorm_obs=False,
    ):
        print("WARNING: This class will soon be deprecated.")
        self.save_init_params(locals())
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.observation_hidden_size = observation_hidden_size
        self.embedded_hidden_size = embedded_hidden_size
        self.hidden_init = hidden_init
        self.obs_fc = nn.Linear(obs_dim, observation_hidden_size)
        self.embedded_fc = nn.Linear(observation_hidden_size + action_dim,
                                     embedded_hidden_size)

        self.last_fc = nn.Linear(embedded_hidden_size, 1)
        self.output_activation = output_activation

        self.init_weights(init_w)
        self.batchnorm_obs = batchnorm_obs
        if self.batchnorm_obs:
            self.bn_obs = nn.BatchNorm1d(obs_dim)

    def init_weights(self, init_w):
        self.hidden_init(self.obs_fc.weight)
        self.obs_fc.bias.data.fill_(0)
        self.hidden_init(self.embedded_fc.weight)
        self.embedded_fc.bias.data.fill_(0)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, obs, action):
        if self.batchnorm_obs:
            obs = self.bn_obs(obs)
        h = obs
        h = F.relu(self.obs_fc(h))
        h = torch.cat((h, action), dim=1)
        h = F.relu(self.embedded_fc(h))
        return self.output_activation(self.last_fc(h))


class FeedForwardPolicy(PyTorchModule):
    def __init__(
            self,
            obs_dim,
            action_dim,
            fc1_size,
            fc2_size,
            init_w=1e-3,
            hidden_init=ptu.fanin_init,
    ):
        print("WARNING: This class will soon be deprecated.")
        self.save_init_params(locals())
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.fc1_size = fc1_size
        self.fc2_size = fc2_size
        self.hidden_init = hidden_init

        self.fc1 = nn.Linear(obs_dim, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.last_fc = nn.Linear(fc2_size, action_dim)

        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.hidden_init(self.fc1.weight)
        self.fc1.bias.data.fill_(0)
        self.hidden_init(self.fc2.weight)
        self.fc2.bias.data.fill_(0)

        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, obs):
        h = F.relu(self.fc1(obs))
        h = F.relu(self.fc2(h))
        return F.tanh(self.last_fc(h))

    def get_action(self, obs_np):
        actions = self.get_actions(obs_np[None])
        return actions[0, :], {}

    def get_actions(self, obs):
        return self.eval_np(obs)


"""
Random Networks Below
"""


class OuterProductFF(PyTorchModule):
    """
    An interesting idea that I had where you first take the outer product of
    all inputs, flatten it, and then pass it through a linear layer. I
    haven't really tested this, but I'll leave it here to tempt myself later...
    """

    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            hidden_init=ptu.fanin_init,
            b_init_value=0.,
    ):
        self.save_init_params(locals())
        super().__init__()

        self.sops = []
        in_size = input_size
        for i, next_size in enumerate(hidden_sizes):
            sop = SelfOuterProductLinear(in_size, next_size)
            in_size = next_size
            hidden_init(sop.fc.weight)
            sop.fc.bias.data.fill_(b_init_value)
            self.__setattr__("sop{}".format(i), sop)
            self.sops.append(sop)
        self.output_activation = output_activation
        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.fill_(b_init_value)

    def forward(self, input, return_preactivations=False):
        h = input
        for i, sop in enumerate(self.sops):
            h = self.hidden_activation(sop(h))
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output
