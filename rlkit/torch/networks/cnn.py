import numpy as np
import torch
from torch import nn as nn

from rlkit.policies.base import Policy
from rlkit.pythonplusplus import identity
from rlkit.torch.core import PyTorchModule, eval_np
from rlkit.torch.data_management.normalizer import TorchFixedNormalizer
from rlkit.torch.pytorch_util import activation_from_string
import torch.nn.functional as F

import numpy as np



class CNN(nn.Module):
    # TODO: remove the FC parts of this code
    def __init__(
            self,
            input_width,
            input_height,
            input_channels,
            output_size,
            kernel_sizes,
            n_channels,
            strides,
            paddings,
            hidden_sizes=None,
            added_fc_input_size=0,
            conv_normalization_type='none',
            fc_normalization_type='none',
            init_w=1e-4,
            hidden_init=nn.init.xavier_uniform_,
            hidden_activation=nn.ReLU(),
            output_activation=identity,
            output_conv_channels=False,
            pool_type='none',
            pool_sizes=None,
            pool_strides=None,
            pool_paddings=None,
            image_augmentation=False,
            image_augmentation_padding=4,
            fc_dropout=0.0,
            fc_dropout_length=0,
    ):
        if hidden_sizes is None:
            hidden_sizes = []
        assert len(kernel_sizes) == \
               len(n_channels) == \
               len(strides) == \
               len(paddings)
        assert conv_normalization_type in {'none', 'batch', 'layer'}
        assert fc_normalization_type in {'none', 'batch', 'layer'}
        assert pool_type in {'none', 'max2d'}
        if pool_type == 'max2d':
            assert len(pool_sizes) == len(pool_strides) == len(pool_paddings)
        super().__init__()

        self.hidden_sizes = hidden_sizes
        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels
        self.output_size = output_size
        self.output_activation = output_activation
        self.hidden_activation = hidden_activation
        self.conv_normalization_type = conv_normalization_type
        self.fc_normalization_type = fc_normalization_type
        self.added_fc_input_size = added_fc_input_size
        self.conv_input_length = self.input_width * self.input_height * self.input_channels
        self.output_conv_channels = output_conv_channels
        self.pool_type = pool_type
        self.image_augmentation = image_augmentation
        self.image_augmentation_padding = image_augmentation_padding
        self.fc_dropout = fc_dropout
        self.fc_dropout_length = fc_dropout_length
        self.hidden_sizes = hidden_sizes
        self.init_w = init_w

        self.conv_layers = nn.ModuleList()
        self.conv_norm_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()
        self.fc_norm_layers = nn.ModuleList()

        for i, (out_channels, kernel_size, stride, padding) in enumerate(
                zip(n_channels, kernel_sizes, strides, paddings)
        ):
            conv = nn.Conv2d(input_channels,
                             out_channels,
                             kernel_size,
                             stride=stride,
                             padding=padding)
            hidden_init(conv.weight)
            conv.bias.data.fill_(0)

            conv_layer = conv
            self.conv_layers.append(conv_layer)
            input_channels = out_channels

            if pool_type == 'max2d':
                if pool_sizes[i] > 1:
                    self.pool_layers.append(
                        nn.MaxPool2d(
                            kernel_size=pool_sizes[i],
                            stride=pool_strides[i],
                            padding=pool_paddings[i],
                        )
                    )

        # use torch rather than ptu because initially the model is on CPU
        test_mat = torch.zeros(
            1,
            self.input_channels,
            self.input_width,
            self.input_height,
        )
        # find output dim of conv_layers by trial and add norm conv layers
        for i, conv_layer in enumerate(self.conv_layers):
            test_mat = conv_layer(test_mat)
            if self.conv_normalization_type == 'batch':
                self.conv_norm_layers.append(nn.BatchNorm2d(test_mat.shape[1]))
            if self.conv_normalization_type == 'layer':
                self.conv_norm_layers.append(nn.LayerNorm(test_mat.shape[1:]))
            if self.pool_type != 'none' and len(self.pool_layers) > i:
                test_mat = self.pool_layers[i](test_mat)

        self.conv_output_flat_size = int(np.prod(test_mat.shape))

        if self.output_conv_channels:
            self.last_fc = None
        else:
            self.fc_layers, self.fc_norm_layers, self.last_fc = self.initialize_fc_layers(self.hidden_sizes, 
                self.output_size, self.conv_output_flat_size, self.added_fc_input_size, init_w)

        if self.image_augmentation:
            self.augmentation_transform = RandomCrop(
                input_height, self.image_augmentation_padding, device='cuda')

        if self.fc_dropout > 0.0:
            self.fc_dropout_layer = nn.Dropout(self.fc_dropout)


    def initialize_fc_layers(self, hidden_sizes, output_size, conv_output_flat_size, added_fc_input_size, init_w):
        fc_layers = nn.ModuleList()
        fc_norm_layers = nn.ModuleList()
    
        
        fc_input_size = conv_output_flat_size
        # used only for injecting input directly into fc layers
        fc_input_size += added_fc_input_size
        for idx, hidden_size in enumerate(hidden_sizes):
            fc_layer = nn.Linear(fc_input_size, hidden_size)
            fc_input_size = hidden_size

            fc_layer.weight.data.uniform_(-init_w, init_w)
            fc_layer.bias.data.uniform_(-init_w, init_w)

            fc_layers.append(fc_layer)

            if self.fc_normalization_type == 'batch':
                fc_norm_layers.append(nn.BatchNorm1d(hidden_size))
            if self.fc_normalization_type == 'layer':
                fc_norm_layers.append(nn.LayerNorm(hidden_size))

        last_fc = nn.Linear(fc_input_size, output_size)
        last_fc.weight.data.uniform_(-init_w, init_w)
        last_fc.bias.data.uniform_(-init_w, init_w)

        return fc_layers, fc_norm_layers, last_fc

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
            h = torch.cat((extra_fc_input, h), dim=1)
        h = self.apply_forward_fc(h)

        if return_last_activations:
            return h
        return self.output_activation(self.last_fc(h))

    def apply_forward_conv(self, h):
        for i, layer in enumerate(self.conv_layers):
            h = layer(h)
            if self.conv_normalization_type != 'none':
                h = self.conv_norm_layers[i](h)
            if self.pool_type != 'none' and len(self.pool_layers) > i:
                h = self.pool_layers[i](h)
            h = self.hidden_activation(h)
        return h

    def apply_forward_fc(self, h):
        if self.fc_dropout > 0.0 and self.fc_dropout_length > 0:
            dropout_input = h.narrow(
                start=0,
                length=self.fc_dropout_length,
                dim=1,
            )
            dropout_output = self.fc_dropout_layer(dropout_input)

            remaining_input = h.narrow(
                start=self.fc_dropout_length,
                length=self.conv_output_flat_size + self.added_fc_input_size - self.fc_dropout_length,
                dim=1)
            h = torch.cat((dropout_output, remaining_input), dim=1)

        for i, layer in enumerate(self.fc_layers):
            h = layer(h)
            if self.fc_normalization_type != 'none':
                h = self.fc_norm_layers[i](h)
            h = self.hidden_activation(h)
        return h


class ConcatCNN(CNN):
    """
    Concatenate inputs along dimension and then pass through MLP.
    """
    def __init__(self, *args, dim=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=self.dim)
        return super().forward(flat_inputs, **kwargs)


class MergedCNN(CNN):
    '''
    CNN that supports input directly into fully connected layers
    '''

    def __init__(self,
                 added_fc_input_size,
                 **kwargs
                 ):
        super().__init__(added_fc_input_size=added_fc_input_size,
                         **kwargs)

    def forward(self, conv_input, fc_input):
        h = torch.cat((conv_input, fc_input), dim=1)
        output = super().forward(h)
        return output


class CNNPolicy(CNN, Policy):
    """
    A simpler interface for creating policies.
    """

    def __init__(
            self,
            *args,
            obs_normalizer: TorchFixedNormalizer = None,
            **kwargs
    ):
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
        return eval_np(self, obs)


class BasicCNN(PyTorchModule):
    # TODO: clean up CNN using this basic CNN
    def __init__(
            self,
            input_width,
            input_height,
            input_channels,
            kernel_sizes,
            n_channels,
            strides,
            paddings,
            normalization_type='none',
            hidden_init=None,
            hidden_activation='relu',
            output_activation=identity,
            pool_type='none',
            pool_sizes=None,
            pool_strides=None,
            pool_paddings=None,
    ):
        assert len(kernel_sizes) == \
               len(n_channels) == \
               len(strides) == \
               len(paddings)
        assert normalization_type in {'none', 'batch', 'layer'}
        assert pool_type in {'none', 'max2d'}
        if pool_type == 'max2d':
            assert len(pool_sizes) == len(pool_strides) == len(pool_paddings)
        super().__init__()

        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels
        self.output_activation = output_activation
        if isinstance(hidden_activation, str):
            hidden_activation = activation_from_string(hidden_activation)
        self.hidden_activation = hidden_activation
        self.normalization_type = normalization_type
        self.conv_input_length = self.input_width * self.input_height * self.input_channels
        self.pool_type = pool_type

        self.conv_layers = nn.ModuleList()
        self.conv_norm_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()

        for i, (out_channels, kernel_size, stride, padding) in enumerate(
                zip(n_channels, kernel_sizes, strides, paddings)
        ):
            conv = nn.Conv2d(input_channels,
                             out_channels,
                             kernel_size,
                             stride=stride,
                             padding=padding)
            if hidden_init:
                hidden_init(conv.weight)

            conv_layer = conv
            self.conv_layers.append(conv_layer)
            input_channels = out_channels

            if pool_type == 'max2d':
                if pool_sizes[i] > 1:
                    self.pool_layers.append(
                        nn.MaxPool2d(
                            kernel_size=pool_sizes[i],
                            stride=pool_strides[i],
                            padding=pool_paddings[i],
                        )
                    )
                else:
                    self.pool_layers.append(None)

        # use torch rather than ptu because initially the model is on CPU
        test_mat = torch.zeros(
            1,
            self.input_channels,
            self.input_height,
            self.input_width,
        )
        # find output dim of conv_layers by trial and add norm conv layers
        for i, conv_layer in enumerate(self.conv_layers):
            test_mat = conv_layer(test_mat)
            if self.normalization_type == 'batch':
                self.conv_norm_layers.append(nn.BatchNorm2d(test_mat.shape[1]))
            if self.normalization_type == 'layer':
                self.conv_norm_layers.append(nn.LayerNorm(test_mat.shape[1:]))
            if self.pool_type != 'none':
                if self.pool_layers[i]:
                    test_mat = self.pool_layers[i](test_mat)

        self.output_shape = test_mat.shape[1:]  # ignore batch dim

    def forward(self, conv_input):
        return self.apply_forward_conv(conv_input)

    def apply_forward_conv(self, h):
        for i, layer in enumerate(self.conv_layers):
            h = layer(h)
            if self.normalization_type != 'none':
                h = self.conv_norm_layers[i](h)
            if self.pool_type != 'none':
                if self.pool_layers[i]:
                    h = self.pool_layers[i](h)
            h = self.hidden_activation(h)
        return h


class RandomCrop:
    """
    Source: # https://github.com/pratogab/batch-transforms
    Applies the :class:`~torchvision.transforms.RandomCrop` transform to
    a batch of images.
    Args:
        size (int): Desired output size of the crop.
        padding (int, optional): Optional padding on each border of the image.
            Default is None, i.e no padding.
        dtype (torch.dtype,optional): The data type of tensors to which the transform will be applied.
        device (torch.device,optional): The device of tensors to which the transform will be applied.
    """

    def __init__(self, size, padding=None, dtype=torch.float, device='cpu'):
        self.size = size
        self.padding = padding
        self.dtype = dtype
        self.device = device

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (N, C, H, W) to be cropped.
        Returns:
            Tensor: Randomly cropped Tensor.
        """
        if self.padding is not None:
            padded = torch.zeros((tensor.size(0), tensor.size(1),
                                  tensor.size(2) + self.padding * 2,
                                  tensor.size(3) + self.padding * 2),
                                 dtype=self.dtype, device=self.device)
            padded[:, :, self.padding:-self.padding,
            self.padding:-self.padding] = tensor
        else:
            padded = tensor

        w, h = padded.size(2), padded.size(3)
        th, tw = self.size, self.size
        if w == tw and h == th:
            i, j = 0, 0
        else:
            i = torch.randint(0, h - th + 1, (tensor.size(0),),
                              device=self.device)
            j = torch.randint(0, w - tw + 1, (tensor.size(0),),
                              device=self.device)

        rows = torch.arange(th, dtype=torch.long, device=self.device) + i[:,
                                                                        None]
        columns = torch.arange(tw, dtype=torch.long, device=self.device) + j[:,
                                                                           None]
        padded = padded.permute(1, 0, 2, 3)
        padded = padded[:, torch.arange(tensor.size(0))[:, None, None],
                 rows[:, torch.arange(th)[:, None]], columns[:, None]]
        return padded.permute(1, 0, 2, 3)
