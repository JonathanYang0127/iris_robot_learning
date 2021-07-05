import torch
import torch.nn as nn
import torch.nn.functional as F

from wide_resnet import Wide_ResNet


def reparameterize(mu, logvar):
    """
    From https://github.com/AntixK/PyTorch-VAE/blob/master/models/beta_vae.py
    Reparameterization trick to sample from N(mu, var) from
    N(0,1).
    :param mu: (Tensor) Mean of the latent Gaussian [B x D]
    :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
    :return: (Tensor) [B x D]
    """
    std = torch.exp(0.5 * logvar)
    # q_z = td.normal.Normal(mu, std)     # create a torch distribution
    eps = torch.randn_like(std)
    z = eps * std + mu
    return z


class EncoderNet(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*9*9, 512)

        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(512, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_var = nn.Linear(256, latent_dim)

    def forward(self, encoder_input):
        t, b, obs_dim = encoder_input.shape
        x = encoder_input.view(t*b, obs_dim)
        x = x.view(t*b, 3, 48, 48)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        z = reparameterize(mu, log_var)
        return z, mu, log_var


class WideResEncoderNet(Wide_ResNet):

    def forward(self, x):
        t, b, obs_dim = x.shape
        x = x.view(t*b, obs_dim)
        x = x.view(t*b, 3, 48, 48)

        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)

        out = self.linear(out)

        mu = out[:, :self.latent_dim]
        log_var = out[:, self.latent_dim:]
        z = reparameterize(mu, log_var)
        return z, mu, log_var


class DecoderNet(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*9*9 + self.latent_dim, 512)

        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2)

    def forward(self, task_embedding, decoder_input):

        t, b, obs_dim = decoder_input.shape
        x = decoder_input.view(t*b, obs_dim)
        x = x.view(t*b, 3, 48, 48)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = torch.cat((x, task_embedding), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class EncoderDecoderNet(nn.Module):
    def __init__(self, latent_dim, encoder_resent=False):
        super().__init__()

        self.latent_dim = latent_dim
        if encoder_resent:
            self.encoder_net = WideResEncoderNet(10, 5, 0.3, latent_dim*2)
        else:
            self.encoder_net = EncoderNet(latent_dim)
        self.decoder_net = DecoderNet(latent_dim)

    def forward(self, encoder_input, decoder_input):
        z, mu, log_var = self.encoder_net(encoder_input)
        predicted_reward = self.decoder_net(z, decoder_input)
        return predicted_reward, mu, log_var