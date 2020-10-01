import numpy as np
from gym.spaces import Box, Dict
from rlkit.core.distribution import DictDistribution
from rlkit.misc.asset_loader import load_local_or_remote_file



class AddLatentDistribution(DictDistribution):
    def __init__(
            self,
            dist,
            input_key,
            output_key,
            model,
    ):
        self.dist = dist
        self._spaces = dist.spaces
        self.input_key = input_key
        self.output_key = output_key
        self.model = model
        self.representation_size = self.model.representation_size
        latent_space = Box(
            -10 * np.ones(self.representation_size),
            10 * np.ones(self.representation_size),
            dtype=np.float32,
        )
        self._spaces[output_key] = latent_space

    def sample(self, batch_size: int, context=None):
        s = self.dist.sample(batch_size)
        s[self.output_key] = self.model.encode_np(s[self.input_key])
        return s

    @property
    def spaces(self):
        return self._spaces

class AddConditionalLatentDistribution(DictDistribution):
    def __init__(
            self,
            dist,
            input_key,
            output_key,
            model,
            context_key="initial_image_observation",
    ):
        self.dist = dist
        self._spaces = dist.spaces
        self.input_key = input_key
        self.output_key = output_key
        self.context_key = context_key
        self.model = model
        self.representation_size = self.model.representation_size
        latent_space = Box(
            -10 * np.ones(self.representation_size),
            10 * np.ones(self.representation_size),
            dtype=np.float32,
        )
        self._spaces[output_key] = latent_space

    def sample(self, batch_size: int, context=None):
        s = self.dist.sample(batch_size)
        s[self.output_key] = self.model.encode_np(s[self.input_key], s[self.context_key])
        return s

    @property
    def spaces(self):
        return self._spaces


class PriorDistribution(DictDistribution):
    def __init__(
            self,
            representation_size,
            key,
            dist=None,
    ):
        self._spaces = dist.spaces if dist else {}
        self.key = key
        self.representation_size = representation_size
        self.dist = dist
        latent_space = Box(
            -10 * np.ones(self.representation_size),
            10 * np.ones(self.representation_size),
            dtype=np.float32,
        )
        self._spaces[key] = latent_space

    def sample(self, batch_size: int, context=None):
        s = self.dist.sample(batch_size) if self.dist else {}
        mu, sigma = 0, 1 # sample from prior
        n = np.random.randn(batch_size, self.representation_size)
        s[self.key] = sigma * n + mu
        return s

    @property
    def spaces(self):
        return self._spaces

class PresampledPriorDistribution(DictDistribution):
    def __init__(
            self,
            datapath,
            key,
            dist=None,
    ):
        self._spaces = dist.spaces if dist else {}
        self._presampled_goals = load_local_or_remote_file(datapath)
        self._num_presampled_goals = self._presampled_goals.shape[0]
        self.representation_size = self._presampled_goals.shape[1]
        self.dist = dist
        self.key = key
        latent_space = Box(
            -10 * np.ones(self.representation_size),
            10 * np.ones(self.representation_size),
            dtype=np.float32,
        )
        self._spaces[key] = latent_space


    def sample(self, batch_size: int, context=None):
        s = self.dist.sample(batch_size) if self.dist else {}
        idx = np.random.randint(0, self._num_presampled_goals, batch_size)
        s[self.key] = self._presampled_goals[idx, :]
        return s

    @property
    def spaces(self):
        return self._spaces

class ConditionalPriorDistribution(DictDistribution):
    def __init__(
            self,
            model,
            key,
            dist=None,
    ):
        self.representation_size = model.representation_size
        self._spaces = dist.spaces if dist else {}
        self.model = model
        self.key = key
        self.dist = dist
        latent_space = Box(
            -10 * np.ones(self.representation_size),
            10 * np.ones(self.representation_size),
            dtype=np.float32,
        )
        self._spaces[key] = latent_space

    def sample(self, batch_size, context=None):
        s = self.dist.sample(batch_size) if self.dist else {}
        z_0 = context['initial_latent_state'].reshape(-1, self.representation_size)
        s[self.key] = self.model.sample_prior(batch_size, cond=z_0, image_cond=False)

        return s

    @property
    def spaces(self):
        return self._spaces

class AmortizedPriorDistribution(DictDistribution):
    def __init__(
            self,
            model,
            key,
            dist=None,
            num_presample=1000,
    ):
        self.representation_size = self.model.representation_size
        self._spaces = dist.spaces if dist else {}
        self.num_presample = num_presample
        self.model = model
        self.key = key
        self.dist = dist
        latent_space = Box(
            -10 * np.ones(self.representation_size),
            10 * np.ones(self.representation_size),
            dtype=np.float32,
        )
        self._spaces[key] = latent_space
        self.presampled_z = self.model.sample_prior(num_presample)
        self.z_index = 0

    def sample(self, batch_size: int, context=None):
        s = self.dist.sample(batch_size) if self.dist else {}
        if batch_size + self.z_index >= self.num_presample:
            self.presampled_z = self.model.sample_prior(self.num_presample)
            self.z_index = 0
        
        s[self.key] = self.presampled_z[self.z_index:self.z_index + batch_size]
        self.z_index += batch_size
        return s

    @property
    def spaces(self):
        return self._spaces

class AmortizedConditionalPriorDistribution(DictDistribution):
    def __init__(
            self,
            model,
            key,
            dist=None,
            num_presample=1000,
    ):
        self.representation_size = model.representation_size
        self.num_presample = num_presample
        self._spaces = dist.spaces if dist else {}
        self.model = model
        self.key = key
        self.dist = dist
        latent_space = Box(
            -10 * np.ones(self.representation_size),
            10 * np.ones(self.representation_size),
            dtype=np.float32,
        )
        self._spaces[key] = latent_space
        self._spaces['initial_latent_state'] = latent_space
        self.z_index = self.num_presample # We need to wait for x_0 to sample goals
        self.presampled_z = None

    def sample(self, batch_size, context=None):
        s = self.dist.sample(batch_size) if self.dist else {}
        x_0 = context['initial_latent_state'].reshape(-1, self.representation_size)
        
        if batch_size + self.z_index >= self.num_presample:
            self.presampled_z = self.model.sample_prior(
                self.num_presample,
                cond=x_0[0].reshape(-1, self.representation_size)
            )
            self.z_index = 0
        
        s[self.key] = self.presampled_z[self.z_index:self.z_index + batch_size]
        self.z_index += batch_size
    
        return s

    @property
    def spaces(self):
        return self._spaces
