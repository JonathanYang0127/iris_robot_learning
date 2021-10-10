import numpy as np
import copy
from sklearn.mixture import GaussianMixture
from rlkit.exploration_strategies.embedding_base_exploration_strategy import BaseExplorationStrategy
import torch
import rlkit.torch.pytorch_util as ptu
import matplotlib
matplotlib.use("PDF")
import matplotlib.pyplot as plt

class CEMExplorationStrategy(BaseExplorationStrategy):
    def __init__(self, embeddings, policy=None, q_function=None, n_components=10, update_frequency=20, update_window=100):
        super().__init__(embeddings, policy=policy, q_function=q_function)
        self.n_components = n_components
        self.update_frequency = update_frequency
        self.update_window=update_window
        self._update_counter = {'forward': 0, 'reverse': 0}
        self._iteration = {'forward': 0, 'reverse': 0}
        self._current_embedding = None
        self._positive_embeddings = {'forward': [], 'reverse': []}

        self.gms = {'forward': self.fit_gaussian(self.embeddings_batch, n_components=self.n_components),
            'reverse': self.fit_gaussian(self.embeddings_batch, n_components=self.n_components)}

    def sample_embedding(self, **kwargs):
        assert 'reverse' in kwargs
        gm_key = 'reverse' if kwargs['reverse'] else 'forward'
        z, _ = self.gms[gm_key].sample()
        self._current_embedding = z.flatten()
        return self._current_embedding

    def post_trajectory_update(self, plot=False, **kwargs):
        assert 'success' in kwargs and 'reverse' in kwargs
        gm_key = 'reverse' if kwargs['reverse'] else 'forward'
        self._update_counter[gm_key] += 1
        if kwargs['success']:
            self._positive_embeddings[gm_key].append(copy.deepcopy(self._current_embedding))
        if self._update_counter[gm_key] % self.update_frequency == 0 and \
            len(self._positive_embeddings[gm_key]) >= self.n_components:
            print("ADAPTING...")
            self._iteration[gm_key] += 1
            self._update_counter[gm_key] = 0
            self.gms[gm_key] = self.fit_gaussian(np.array(self._positive_embeddings[gm_key][-self.update_window:]),
                n_components=self.n_components, plot=plot)

    def fit_gaussian(self, batch, n_components=1, plot=False):
        gm = GaussianMixture(n_components=n_components)
        gm = gm.fit(batch)

        if plot:
            plt.figure()
            plt.scatter(self.embeddings_batch[:, 0], self.embeddings_batch[:, 1], c='grey', s=0.1)
            plt.scatter(batch[:, 0], batch[:, 1], c='green')
            plt.scatter(gm.means_[:, 0], gm.means_[:, 1], c='black')
            plt.savefig('plot_embeddings.png')

        return gm
