import numpy as np
import torch
from gym.spaces import Box

from rlkit.core import logger
from rlkit.core.distribution import DictDistribution
from rlkit.core.distribution import DictDistributionGenerator
from rlkit.util.io import load_local_or_remote_file
from rlkit.torch import pytorch_util as ptu

class AddDecodedImageDistribution(DictDistribution):
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
        self.image_size = self.model.imlength
        image_space = Box(
            np.zeros(self.image_size),
            np.ones(self.image_size),
            dtype=np.float32,
        )
        self._spaces[output_key] = image_space

    def sample(self, batch_size: int):
        s = self.dist.sample(batch_size)
        s[self.output_key] = self.model.decode_np(
            s[self.input_key]).reshape((-1, self.image_size))
        return s

    def __call__(self, context):
        self.dist.context = context
        return self

    @property
    def spaces(self):
        return self._spaces


class AddGripperStateDistribution(DictDistribution):
    def __init__(
        self,
        dist,
        input_key,
        output_key,
    ):
        self.dist = dist
        self._spaces = dist.spaces
        self.input_key = input_key
        self.output_key = output_key
        self.gripper_state_size = 6
        gripper_state_space = Box(
            -1 * np.ones(self.gripper_state_size),
            1 * np.ones(self.gripper_state_size),
            dtype=np.float32,
        )
        self._spaces[output_key] = gripper_state_space

    def sample(self, batch_size: int):
        from roboverse.bullet.misc import quat_to_deg_batch
        s = self.dist.sample(batch_size)
        s[self.output_key] = np.concatenate(
            (s[self.input_key][:, :3],
             quat_to_deg_batch(s[self.input_key][:, 3:7]) / 360.),
            axis=1)
        return s

    def __call__(self, context):
        self.context = self.dist(context)
        return self

    @property
    def spaces(self):
        return self._spaces


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

    def sample(self, batch_size: int):
        s = self.dist.sample(batch_size)
        s[self.output_key] = self.model.encode_np(s[self.input_key])
        return s

    def __call__(self, context):
        self.context = self.dist(context)
        return self

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

    def sample(self, batch_size: int):
        s = self.dist.sample(batch_size)
        s[self.output_key] = self.model.encode_np(
            s[self.input_key], s[self.context_key])
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

    def sample(self, batch_size: int):
        s = self.dist.sample(batch_size) if self.dist else {}
        mu, sigma = 0, 1  # sample from prior
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

    def sample(self, batch_size: int):
        s = self.dist.sample(batch_size) if self.dist else {}
        idx = np.random.randint(0, self._num_presampled_goals, batch_size)
        s[self.key] = self._presampled_goals[idx, :]
        return s

    @property
    def spaces(self):
        return self._spaces


class PresamplePriorDistribution(DictDistribution):
    def __init__(
            self,
            model,
            key,
            env,
            num_presample=5000,
            samples_per_batch=50,
            dist=None,
            save_goals=False,
    ):
        self.representation_size = model.representation_size
        self._spaces = dist.spaces if dist else {}
        self.model = model
        self.key = key
        self.env = env
        self.num_batches = num_presample // samples_per_batch
        self.samples_per_batch = samples_per_batch
        self.dist = dist
        self._presampled_goals = self.presample_goals()

        if save_goals:
            logger.save_extra_data(
                self._presampled_goals, "presampled_goals", "pickle")

        self._num_presampled_goals = self._presampled_goals.shape[0]
        latent_space = Box(
            -10 * np.ones(self.representation_size),
            10 * np.ones(self.representation_size),
            dtype=np.float32,
        )
        self._spaces[key] = latent_space

    def presample_goals(self):
        dataset = []
        for i in range(self.num_batches):
            self.env.reset()
            init_obs = ptu.from_numpy(
                np.uint8(self.env.render_obs()).transpose() / 255.0)
            init_z = ptu.get_numpy(self.model.encode(init_obs))
            sampled_z = self.model.sample_prior(
                self.samples_per_batch, cond=init_z)
            dataset.append(sampled_z)
        return np.concatenate(dataset, axis=0)

    def sample(self, batch_size: int):
        s = self.dist.sample(batch_size) if self.dist else {}
        idx = np.random.randint(0, self._num_presampled_goals, batch_size)
        s[self.key] = self._presampled_goals[idx, :]
        return s

    @property
    def spaces(self):
        return self._spaces


class ConditionalPriorDistribution(DictDistributionGenerator):
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
        self.context = None

    def sample(self, batch_size):
        s = self.dist.sample(batch_size) if self.dist else {}
        z_0 = self.context['initial_latent_state'].reshape(
            -1, self.representation_size)
        s[self.key] = self.model.sample_prior(
            batch_size, cond=z_0)  # , image_cond=False)
        return s

    def __call__(self, context):
        self.context = context
        return self

    @property
    def spaces(self):
        return self._spaces


class CLearningConditionalPriorDistribution(DictDistributionGenerator):
    def __init__(
            self,
            model,
            key,
            dist=None,
            policy=None,
            qf1=None,
            qf2=None,
            pixelcnn_sample_k=10,
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

        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.k = pixelcnn_sample_k

    def sample(self, batch_size):
        s = self.dist.sample(batch_size) if self.dist else {}
        z_0 = self.context['initial_latent_state'].reshape(
            -1, self.representation_size)

        # Faster, but requires more memory
        z_0 = np.repeat(z_0, self.k, axis=0)
        goal = self.model.sample_prior(batch_size*self.k, z_0)
        score = self.score_fn(z_0, goal)
        assert score.shape == (batch_size*self.k,)
        best_goals = np.empty((batch_size, self.representation_size))
        for i, score_i in enumerate(range(0, batch_size*self.k, self.k)):
            idx = score_i+np.argmax(score[score_i:score_i+self.k])
            best_goals[i] = goal[idx]

        # Slower, but requires less memory
        # best_scores = np.zeros((batch_size,))
        # best_goals = np.empty((batch_size, self.representation_size))
        # for _ in range(self.k):
        #     goal = self.model.sample_prior(batch_size, cond=z_0)
        #     score = self.score_fn(z_0, goal)
        #     indices = np.nonzero(score > best_scores)
        #     best_scores[indices] = score[indices]
        #     best_goals[indices] = goal[indices]

        s[self.key] = best_goals
        return s

    def score_fn(self, s_current, s_future):
        obs = ptu.from_numpy(np.concatenate((s_current, s_future), axis=1))
        a_next, log_pi = self.policy(obs).rsample_and_logprob()
        classification = torch.min(
            self.qf1(obs, a_next, ), self.qf2(obs, a_next, ))
        w = classification / (1 - classification)
        return ptu.get_numpy(w).squeeze(axis=1)

    def __call__(self, context):
        self.context = context
        return self

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

    def sample(self, batch_size: int):
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


class AmortizedConditionalPriorDistribution(DictDistributionGenerator):

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

        # We need to wait for x_0 to sample goals
        self.z_index = self.num_presample

        self.presampled_z = None

    def sample(self, batch_size):
        s = self.dist.sample(batch_size) if self.dist else {}
        x_0 = self.context['initial_latent_state'].reshape(
            -1, self.representation_size)

        if batch_size + self.z_index >= self.num_presample:
            self.presampled_z = self.model.sample_prior(
                self.num_presample,
                cond=x_0[0].reshape(-1, self.representation_size)
            )
            self.z_index = 0

        s[self.key] = self.presampled_z[self.z_index:self.z_index + batch_size]
        self.z_index += batch_size

        return s

    def __call__(self, context):
        self.context = context
        return self

    @property
    def spaces(self):
        return self._spaces
