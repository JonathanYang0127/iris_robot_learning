import torch
from torch import nn

from rlkit.torch.core import PyTorchModule


class Flatten(nn.Module):

    def forward(self, input):
        return input.view(input.size(0), -1)


class Classifier(PyTorchModule):

    def __init__(
            self,
            data_channels,
            hidden_dim=64,
            fc_dim=256,
            imsize=12,
            decay=0.0,
    ):
        super(Classifier, self).__init__()

        self.imsize = imsize
        self.hidden_dim = hidden_dim

        self._layer = nn.Sequential(
            nn.Conv2d(
                in_channels=data_channels * 2,
                out_channels=hidden_dim // 2,
                kernel_size=3,
                stride=2,
                padding=1),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=hidden_dim // 2,
                out_channels=hidden_dim,
                kernel_size=3,
                stride=2,
                padding=1),
            nn.ReLU(),

            Flatten(),

            nn.Linear(hidden_dim * 9, fc_dim),
            nn.ReLU(),

            nn.Linear(fc_dim, 1),
        )

    def __call__(self, h0, h1):
        assert h0.shape[-1] == self.imsize
        assert h0.shape[-2] == self.imsize
        assert h1.shape[-1] == self.imsize
        assert h1.shape[-2] == self.imsize

        inputs = torch.cat((h0, h1), dim=1)
        logits = self._layer(inputs)
        return logits


class Discriminator(PyTorchModule):

    def __init__(
            self,
            data_channels,
            hidden_dim=64,
            imsize=12,
            decay=0.0,
    ):
        super(Discriminator, self).__init__()

        self.imsize = imsize

        self.hidden_dim = hidden_dim

        self._layer = nn.Sequential(
            nn.Conv2d(
                in_channels=data_channels * 2,
                out_channels=hidden_dim // 2,
                kernel_size=3,
                stride=2,
                padding=1),
            nn.LeakyReLU(),
            nn.Dropout(0.3),

            nn.Conv2d(
                in_channels=hidden_dim // 2,
                out_channels=hidden_dim,
                kernel_size=3,
                stride=2,
                padding=1),
            nn.LeakyReLU(),
            nn.Dropout(0.3),

            Flatten(),

            nn.Linear(hidden_dim * 9, 1),
        )

    def __call__(self, h0, h1):
        assert h0.shape[-1] == self.imsize
        assert h0.shape[-2] == self.imsize
        assert h1.shape[-1] == self.imsize
        assert h1.shape[-2] == self.imsize

        # inputs = torch.cat((h0, h1), dim=1)
        inputs = torch.cat((h1 - h0, h1), dim=1)
        logits = self._layer(inputs)
        return logits
