import torch
from torch import nn as nn
import torch.nn.modules.conv as conv
import torch.nn.functional as F

import numpy as np
from rlkit.torch.networks.cnn import CNN

class VQVAEWrapper(CNN):
    """
    Wrapper for VQVAE observation processor.
    """
    def __init__(self, vqvae, *args, encoding_type='e', **kwargs):
        super().__init__(*args, **kwargs)
        self.vqvae = vqvae
        self.encoding_type = encoding_type
        self.conv_layers = None
        self.conv_norm_layers = None
        self.pool_layers = None

        for param in self.vqvae.parameters():
            param.requires_grad = False

        test_mat = torch.zeros(
            1,
            self.input_channels,
            self.input_width,
            self.input_height,
        ).cuda()
        test_mat = self.apply_forward_conv(test_mat)
        self.conv_output_flat_size = int(np.prod(test_mat.shape))
        self.fc_layers, self.fc_norm_layers, self.last_fc = self.initialize_fc_layers(self.hidden_sizes,
            self.output_size, self.conv_output_flat_size, self.added_fc_input_size, self.init_w)


    def apply_forward_conv(self, h):
        z_e = self.vqvae.encoder(h)
        z_e = self.vqvae.pre_quantization_conv(z_e)
        if self.encoding_type == 'e':
            return z_e

        embedding_loss, z_q, perplexity, _, _ = self.vqvae.vector_quantization(
            z_e)

        return z_q


    def forward(self, input, return_last_activations=False, precomputed_encodings=False):
        if self.conv_input_length + self.added_fc_input_size == input.shape[1]:
            precomputed_encodings = False
        elif self.conv_output_flat_size + self.added_fc_input_size == input.shape[1]:
            precomputed_encodings = True
        else:
            raise NotImplementedError

        conv_input_length = self.conv_input_length
        if precomputed_encodings:
            conv_input_length = self.conv_output_flat_size

        conv_input = input.narrow(start=0,
                                  length=conv_input_length,
                                  dim=1).contiguous()

        if not precomputed_encodings:
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
        else:
            h = conv_input

        if self.added_fc_input_size != 0:
            extra_fc_input = input.narrow(
                start=conv_input_length,
                length=self.added_fc_input_size,
                dim=1,
            )
            h = torch.cat((extra_fc_input, h), dim=1)
        h = self.apply_forward_fc(h)

        if return_last_activations:
            return h
        return self.output_activation(self.last_fc(h))


class ConcatVQVAEWrapper(VQVAEWrapper):
    """
    Concatenate inputs along dimension and then pass through MLP.
    """
    def __init__(self, *args, dim=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=self.dim)
        return super().forward(flat_inputs, **kwargs)


class PrecomputedEncodings(nn.Module):
    """
    Concatenate inputs along dimension and then pass through MLP.
    """
    def __init__(self, conv_network, *args, **kwargs):
        assert not kwargs.get('output_conv_channels', False)
        super().__init__(*args, **kwargs)
        self.conv_network = conv_network

    def forward(self, *inputs, **kwargs):
        kwargs['precomputed_encodings'] = True
        return self.conv_network(*inputs, **kwargs)


