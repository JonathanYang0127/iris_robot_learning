from __future__ import print_function

# import numpy as np  # NOQA
# import torch
# import torch.utils.data
from torch import nn
from torch.nn import functional as F
from rlkit.torch import pytorch_util as ptu  # NOQA


class Residual(nn.Module):

    def __init__(self,
                 in_channels,
                 residual_hidden_dim,
                 out_activation='relu'):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=residual_hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=residual_hidden_dim,
                out_channels=in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
        )
        self._out_activation = out_activation

    def forward(self, x):
        if self._out_activation == 'relu':
            return F.relu(x + self._block(x))
        else:
            return x + self._block(x)


class ResidualStack(nn.Module):

    def __init__(self,
                 in_channels,
                 num_residual_layers,
                 residual_hidden_dim,
                 out_activation='relu'):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers

        layers = []
        for i in range(self._num_residual_layers):
            if i == self._num_residual_layers - 1:
                out_activation_i = out_activation
            else:
                out_activation_i = 'relu'

            layer = Residual(in_channels=in_channels,
                             residual_hidden_dim=residual_hidden_dim,
                             out_activation=out_activation_i)
            layers.append(layer)

        self._layers = nn.ModuleList(layers)

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return x
