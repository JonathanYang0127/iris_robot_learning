import torch
from torch import nn as nn
import torch.nn.modules.conv as conv
import torch.nn.functional as F

from rlkit.pythonplusplus import identity
from rlkit.torch.networks.cnn import RandomCrop

import numpy as np

class IMPALACNN(nn.Module):
    def __init__(
            self, 
            input_width,
            input_height,
            input_channels,
            output_size,
            kernel_sizes,
            n_channels,
            *args,
            hidden_sizes=None,
            added_fc_input_size=0,
            init_w=1e-4,
            hidden_init=nn.init.xavier_uniform_,
            hidden_activation=nn.ReLU(),
            output_activation=identity,
            output_conv_channels=False,
            image_augmentation=False, 
            image_augmentation_padding=4,
            **kwargs
        ):
        if hidden_sizes is None:
            hidden_sizes = []
        assert len(kernel_sizes) == \
               len(n_channels)

        print("The following CNN parameters are not being used: ", list(kwargs.keys()))
        super().__init__()
        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        self.n_channels = n_channels
        self.output_size = output_size
        self.kernel_sizes = kernel_sizes
        self.image_augmentation = image_augmentation
        self.image_augmentation_padding = image_augmentation_padding
        self.conv_input_length = self.input_width * self.input_height * self.input_channels
        self.hidden_sizes = hidden_sizes
        self.init_w = init_w
        self.added_fc_input_size = added_fc_input_size
        self.output_conv_channels = output_conv_channels

        self.conv_params = []
        self.conv_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()
        self.construct_cnn()
        self.construct_fc_layers()

        if self.image_augmentation:
            self.augmentation_transform = RandomCrop(
                input_height, self.image_augmentation_padding, device='cuda')

    def residual_block(self, channels, kernel_size=3):
        if kernel_size % 2 == 0:
            same_padding = ((kernel_size - 1) // 2, (kernel_size - 1) // 2 + 1)
        else:
            same_padding = (kernel_size - 1) // 2
        self.conv_params.append("start")
        self.conv_params.append(self.hidden_activation)

        self.conv_params.append(nn.Conv2d(channels, channels, 
            kernel_size, padding=same_padding))
        self.conv_params.append(self.hidden_activation)
        self.conv_params.append(nn.Conv2d(channels, channels, kernel_size,
            padding=same_padding))
        self.conv_params.append("end")

    def conv_sequence(self, input_dim, channels, kernel_size):
        self.conv_params.append(nn.Conv2d(input_dim, channels, 
            kernel_size, padding=1))
        self.conv_params.append(nn.MaxPool2d(3, 2, padding=1))
        self.residual_block(channels)
        self.residual_block(channels)
        return input

    def construct_cnn(self):
        for i in range(len(self.n_channels)):
            if i == 0:
                self.conv_sequence(self.input_channels, self.n_channels[i], self.kernel_sizes[i])
            else:
                self.conv_sequence(self.n_channels[i - 1], self.n_channels[i], self.kernel_sizes[i])

        self.conv_layers.extend([p for p in self.conv_params if isinstance(p, nn.Module)])

    def construct_fc_layers(self):
        test_mat = test_mat = torch.zeros(
            1,
            self.input_channels,
            self.input_width,
            self.input_height,
        )

        for conv_layer in self.conv_layers:
            test_mat = conv_layer(test_mat)

        self.conv_output_flat_size = int(np.prod(test_mat.shape))

        fc_input_size = self.conv_output_flat_size
        # used only for injecting input directly into fc layers
        fc_input_size += self.added_fc_input_size
        for idx, hidden_size in enumerate(self.hidden_sizes):
            fc_layer = nn.Linear(fc_input_size, hidden_size)
            fc_input_size = hidden_size

            fc_layer.weight.data.uniform_(-self.init_w, self.init_w)
            fc_layer.bias.data.uniform_(-self.init_w, self.init_w)

            self.fc_layers.append(fc_layer)

        self.last_fc = nn.Linear(fc_input_size, self.output_size)
        self.last_fc.weight.data.uniform_(-self.init_w, self.init_w)
        self.last_fc.bias.data.uniform_(-self.init_w, self.init_w)

    def forward(self, input, return_last_activations=False):
        conv_input = input.narrow(start=0,
                                  length=self.conv_input_length,
                                  dim=1).contiguous()
        # reshape from batch of flattened images into (channels, w, h)
        h = conv_input.view(conv_input.shape[0],
                            self.input_channels,
                            self.input_height,
                            self.input_width)

        if h.shape[0] > 1 and self.image_augmentation:
            # h.shape[0] > 1 ensures we apply this only during training
            h = self.augmentation_transform(h)

        h = self.apply_forward_conv(h)

        if self.output_conv_channels:
            return h

        # flatten channels for fc layers
        h = h.view(h.size(0), -1)
        if self.added_fc_input_size != 0:
            extra_fc_input = input.narrow(
                start=self.conv_input_length,
                length=self.added_fc_input_size,
                dim=1,
            )
            h = torch.cat((h, extra_fc_input), dim=1)
        h = self.apply_forward_fc(h)

        if return_last_activations:
            return h
        return self.output_activation(self.last_fc(h))

    def apply_forward_conv(self, h):
        if h.shape[0] > 1 and self.image_augmentation:
            h = self.augmentation_transform(h)

        for p in self.conv_params:
            if isinstance(p, str):
                if p == 'start':
                    start = h
                elif p == 'end':
                    h += start
            else:
                h = p(h)
        return h

    def apply_forward_fc(self, h):
        for layer in self.fc_layers:
            h = layer(h)
            h = self.hidden_activation(h)
        
        return h


class ConcatIMPALACNN(IMPALACNN):
    """
    Concatenate inputs along dimension and then pass through MLP.
    """
    def __init__(self, *args, dim=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=self.dim)
        return super().forward(flat_inputs, **kwargs)

