import numpy as np
import copy
from sklearn.mixture import GaussianMixture
from rlkit.exploration_strategies.embedding_base_exploration_strategy import BaseExplorationStrategy
import torch
import rlkit.torch.pytorch_util as ptu
import matplotlib
matplotlib.use("PDF")
import matplotlib.pyplot as plt

class FastExplorationStrategy(BaseExplorationStrategy):
    def __init__(self, embeddings, policy=None, q_function=None, n_components=10, update_frequency=20,
        epsilon0=0.3):
        super().__init__(embeddings, policy=policy, q_function=q_function)
        self.n_components = n_components
        self.update_frequency = update_frequency
        self._update_counter = {'forward': 0, 'reverse': 0}
        self._iteration = {'forward': 0, 'reverse': 0}
        self._current_embedding = None
        self._positive_embeddings = {'forward': [], 'reverse': []}
        self._embedding_idx = {'forward': None, 'reverse': None}
        self._embedding_probs = {'forward': [], 'reverse': []}
        self.epsilon0 = epsilon0
        self.epsilon = epsilon0

        self.gms = {'forward': self.fit_gaussian(self.embeddings_batch, n_components=self.n_components),
            'reverse': self.fit_gaussian(self.embeddings_batch, n_components=self.n_components)}

    def sample_embedding(self, **kwargs):
        assert 'reverse' in kwargs
        gm_key = 'reverse' if kwargs['reverse'] else 'forward'
        use_random_embedding = bool(np.random.rand() < self.epsilon)
        if (use_random_embedding or len(self._positive_embeddings[gm_key]) <= 1) and len(self._positive_embeddings[gm_key]) <= 40:
            z, _ = self.gms[gm_key].sample()
            self._current_embedding = z.flatten()
            self._embedding_idx[gm_key] = None
        else:
            probs = np.array(self._embedding_probs[gm_key]) / np.sum(self._embedding_probs[gm_key])
            self._embedding_idx[gm_key] = np.random.choice(len(self._positive_embeddings[gm_key]),
                p=probs)
            self._current_embedding = self._positive_embeddings[gm_key][self._embedding_idx[gm_key]]
        return self._current_embedding

    def post_trajectory_update(self, plot=False, **kwargs):
        assert 'success' in kwargs and 'reverse' in kwargs
        gm_key = 'reverse' if kwargs['reverse'] else 'forward'
        self._update_counter[gm_key] += 1
        embedding_idx = self._embedding_idx[gm_key]
        if embedding_idx is not None:
            self._embedding_probs[gm_key][embedding_idx] = 0.8 * self._embedding_probs[gm_key][embedding_idx] + 0.2 * int(kwargs['success'])
        else:
            if kwargs['success']:
                if 'embedding' in kwargs:
                    update_embedding = kwargs['embedding']
                else:
                    update_embedding = self._current_embedding
                self._positive_embeddings[gm_key].append(copy.deepcopy(update_embedding))
                self._embedding_probs[gm_key].append(1.0)
  
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

