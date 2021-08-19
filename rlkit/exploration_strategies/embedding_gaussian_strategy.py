import numpy as np
from sklearn.mixture import GaussianMixture
from rlkit.exploration_strategies.embedding_base_exploration_strategy import BaseExplorationStrategy
import torch
import rlkit.torch.pytorch_util as ptu
import matplotlib
matplotlib.use("PDF")
import matplotlib.pyplot as plt

class GaussianExplorationStrategy(BaseExplorationStrategy):
    def __init__(self, embeddings, policy=None, q_function=None, n_components=5):
        super().__init__(embeddings, policy=policy, q_function=q_function)
        self.n_components = n_components
        self.gm = self.fit_gaussian(self.embeddings_batch, n_components=n_components)

    def sample_embedding(self, **kwargs):
        z, _ = self.gm.sample()
        return z.flatten()

    def fit_gaussian(self, batch, n_components=1):
        gm = GaussianMixture(n_components=1)
        gm = gm.fit(batch)
        return gm

    def filter_embeddings_by_q_values(self, obs, percentile=90, plot_embeddings=True):
        embeddings_batch = self.embeddings_batch
        
        obs_image = obs['image'][None].repeat(t*b, axis=0)
        obs_state = obs['state'][None].repeat(t*b, axis=0)

        obs_full = np.concatenate((obs_image, obs_state, embeddings_batch), axis=1)
        actions = self.run_network_batch(self.policy, obs_full, post_func=lambda x: x.sample())
        vf_obs = self.run_network_batch(self.q_function, np.concatenate((obs_full, actions), axis=1))

        cutoff = np.percentile(vf_obs, percentile)
        cutoff_indices, _ = np.where(vf_obs > cutoff)
        embeddings_percentile = embeddings_batch[cutoff_indices]
        self.gm = GaussianMixture(n_components=10)
        self.gm = self.gm.fit(embeddings_percentile)

        if plot_embeddings:
            plt.figure()
            plt.scatter(embeddings_batch[:, 0], embeddings_batch[:, 1], c='grey', s=0.1)
            plt.scatter(embeddings_percentile[:, 0], embeddings_percentile[:, 1], c='green', s=0.1)
            plt.savefig('plot_embeddings.png')
            plt.figure()
            plt.imsave('plot_image.png', obs['image'].reshape(3, 64, 64).transpose(1, 2, 0))


    def filter_embeddings_by_q_values_v2(self, obs, percentile=95, plot_embeddings=True):
        embeddings_batch = np.array([self.sample_embedding() for i in range(10000)])

        obs_image = obs['image'][None].repeat(10000, axis=0)
        obs_state = obs['state'][None].repeat(10000, axis=0)

        obs = np.concatenate((obs_image, obs_state, embeddings_batch), axis=1)
        actions = self.run_network_batch(self.policy, obs, post_func=lambda x: x.sample())
        vf_obs = self.run_network_batch(self.q_function, np.concatenate((obs, actions), axis=1))

        cutoff = np.percentile(vf_obs, percentile)
        cutoff_indices, _ = np.where(vf_obs > cutoff)
        embeddings_percentile = embeddings_batch[cutoff_indices]
        self.gm = GaussianMixture(n_components=10)
        self.gm = self.gm.fit(embeddings_percentile)

        if plot_embeddings:
            plt.figure()
            plt.scatter(embeddings_batch[:, 0], embeddings_batch[:, 1], c='grey', s=0.1)
            plt.scatter(embeddings_percentile[:, 0], embeddings_percentile[:, 1], c='green', s=0.1)
            plt.savefig('plot_embeddings.png')
