import numpy as np
import copy
from rlkit.exploration_strategies.embedding_base_exploration_strategy import BaseExplorationStrategy
import torch
import rlkit.torch.pytorch_util as ptu
import matplotlib
matplotlib.use("PDF")
import matplotlib.pyplot as plt
from rlkit.core import logger
from scipy import stats

def one_hot(index, length):
    x = np.zeros((length,))
    x[index] = 1.
    return x

class ClosestExplorationStrategy(BaseExplorationStrategy):
    def __init__(self, embeddings, exploration_period=20):
        self._latent_dim = embeddings.shape[1]
        self._exploration_period = exploration_period
        self._positive_embeddings = {'forward': [], 'reverse': []}
        self._fixed_embedding = {'forward': None, 'reverse': None}
        super().__init__(embeddings)

    def sample_embedding(self, **kwargs):
        assert 'reverse' in kwargs
        dir = 'reverse' if kwargs['reverse'] else 'forward'

        if len(self._positive_embeddings[dir]) <= self._exploration_period:
            self._current_embedding = one_hot(np.random.randint(self._latent_dim), self._latent_dim)
        else:
            if self._fixed_embedding[dir] is None:
                mode = stats.mode(np.argmax(np.array(self._positive_embeddings[dir]), axis=1))[0][0]
                self._fixed_embedding[dir] = one_hot(mode, self._latent_dim)
            self._current_embedding = self._fixed_embedding[dir]
        return self._current_embedding

    def post_trajectory_update(self, **kwargs):
        assert 'success' in kwargs and 'reverse' in kwargs
        dir = 'reverse' if kwargs['reverse'] else 'forward'
        if kwargs['success']:
            self._positive_embeddings[dir].append(copy.deepcopy(self._current_embedding))