import torch
import torch.nn as nn
import torch.nn.functional as F

from rlkit.torch.task_encoders.wide_resnet import Wide_ResNet
from rlkit.torch.networks.transformer import SmallGPTConfig, GPT
from rlkit.torch.networks.cnn import RandomCrop


class VanillaEncoderNet(nn.Module):

    def __init__(self, latent_dim, image_size, image_augmentation=False):
        super().__init__()

        self.latent_dim = latent_dim
        self.image_size = image_size
        self.image_augmentation = image_augmentation
        self.augmentation_padding = 4

        if self.image_augmentation:
            self.augmentation_transform = RandomCrop(
                image_size, self.augmentation_padding, device='cuda')

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        if image_size == 64:
            self.fc1 = nn.Linear(16 * 13 * 13, 120)
        elif image_size == 48:
            self.fc1 = nn.Linear(16 * 9 * 9, 120)
        else:
            raise ValueError
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.latent_dim)

    def forward(self, x):
        batch_size, obs_dim = x.shape
        conv_input = x.narrow(start=0, length=3*self.image_size*self.image_size,
                              dim=1).contiguous()
        x = conv_input.view(batch_size, 3, self.image_size, self.image_size)

        if x.shape[0] > 1 and self.image_augmentation:
            # x.shape[0] > 1 ensures we apply this only during training
            x = self.augmentation_transform(x)

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


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
    def __init__(self, image_size, latent_dim, image_augmentation=False,
                 augmentation_padding=4):

        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.image_augmentation = image_augmentation
        self.augmentation_padding = augmentation_padding

        if self.image_augmentation:
            self.augmentation_transform = RandomCrop(
                image_size, self.augmentation_padding, device='cuda')

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        if self.image_size == 48:
            # (48 - 4)/2 (22 - 4 / 2) = 9
            flat_dim = 16*9*9
        elif self.image_size == 64:
            # (64 - 4)/2 (30 - 4) / 2 = 13
            flat_dim = 16*13*13
        else:
            raise ValueError

        self.fc1 = nn.Linear(flat_dim, 512)

        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(512, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_var = nn.Linear(256, latent_dim)

    def forward(self, encoder_input):
        t, b, obs_dim = encoder_input.shape
        x = encoder_input.view(t*b, obs_dim)
        x = x.view(t*b, 3, self.image_size, self.image_size)

        if x.shape[0] > 1 and self.image_augmentation:
            # x.shape[0] > 1 ensures we apply this only during training
            x = self.augmentation_transform(x)

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
        x = x.view(t*b, 3, self.image_size, self.image_size)

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

class TransformerEncoderNet(nn.Module):
    def __init__(self, image_size, latent_dim, image_augmentation=False,
                 augmentation_padding=4, encoder_keys=['observations']):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        
        self.config = SmallGPTConfig(2, 15)
        self.config.cnn_params['image_augmentation'] = image_augmentation
        assert encoder_keys[0] == 'observations'
        self.config.encoder_keys = encoder_keys
        self.encoder = GPT(self.config) 

    def forward(self, *encoder_input):
        num_tasks, b, num_timesteps, _ = encoder_input[0].shape
        encoder_input = [e.view(num_tasks * b, num_timesteps, -1) for e in encoder_input]
        out = self.encoder(*encoder_input)
        mu = out[:, :self.latent_dim]
        log_var = out[:, self.latent_dim:]
        z = reparameterize(mu, log_var)
        return z, mu, log_var

class DecoderNet(nn.Module):
    def __init__(self, image_size, latent_dim, image_augmentation=False,
                 augmentation_padding=4):
        super().__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.image_augmentation = image_augmentation
        self.augmentation_padding = augmentation_padding

        if self.image_augmentation:
            self.augmentation_transform = RandomCrop(
                image_size, self.augmentation_padding, device='cuda')

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        if self.image_size == 48:
            # (48 - 4)/2 (22 - 4 / 2) = 9
            flat_dim = 16*9*9
        elif self.image_size == 64:
            # (64 - 4)/2 (30 - 4) / 2 = 13
            flat_dim = 16*13*13
        else:
            raise ValueError
        self.fc1 = nn.Linear(flat_dim + self.latent_dim, 512)

        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2)

    def forward(self, task_embedding, decoder_input):

        t, b, obs_dim = decoder_input.shape
        x = decoder_input.view(t*b, obs_dim)
        x = x.view(t*b, 3, self.image_size, self.image_size)

        if x.shape[0] > 1 and self.image_augmentation:
            # h.shape[0] > 1 ensures we apply this only during training
            x = self.augmentation_transform(x)

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = torch.cat((x, task_embedding), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class EncoderDecoderNet(nn.Module):
    def __init__(self, image_size, latent_dim, image_augmentation=False,
                 encoder_resnet=False):
        super().__init__()

        self.latent_dim = latent_dim
        self.image_augmentation = image_augmentation

        if encoder_resnet:
            self.encoder_net = WideResEncoderNet(image_size, 10, 5, 0.3, latent_dim*2)
        else:
            self.encoder_net = EncoderNet(image_size,
                                          latent_dim,
                                          image_augmentation=image_augmentation)
        self.decoder_net = DecoderNet(image_size,
                                      latent_dim,
                                      image_augmentation=image_augmentation)

    def forward(self, encoder_input, decoder_input):
        z, mu, log_var = self.encoder_net(encoder_input)
        predicted_reward = self.decoder_net(z, decoder_input)
        return predicted_reward, mu, log_var


class TransformerEncoderDecoderNet(nn.Module):
    def __init__(self, image_size, latent_dim, image_augmentation=False, 
        encoder_keys=['observations']):
        super().__init__()

        self.latent_dim = latent_dim
        self.image_augmentation = image_augmentation

        self.encoder_net = TransformerEncoderNet(image_size,
                                      latent_dim,
                                      image_augmentation=image_augmentation,
                                      encoder_keys=encoder_keys)
        self.decoder_net = DecoderNet(image_size,
                                      latent_dim,
                                      image_augmentation=image_augmentation)

    def forward(self, encoder_input, decoder_input):
        if isinstance(encoder_input, torch.Tensor):
            z, mu, log_var = self.encoder_net(encoder_input)
        elif isinstance(encoder_input, (list, tuple)):
            z, mu, log_var = self.encoder_net(*encoder_input)
        predicted_reward = self.decoder_net(z, decoder_input)
        return predicted_reward, mu, log_var

