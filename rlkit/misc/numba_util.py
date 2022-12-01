import numpy as np
from numba import jit

@jit(nopython=True)
def concatenate_nb(arrs, axis=0):
    return np.concatenate(arrs, axis=axis)

@jit(nopython=True)
def rand_choice_prob_nb(arr, prob):
    """
    :param arr: A 1D numpy array of values to sample from.
    :param prob: A 1D numpy array of probabilities for the given samples.
    :return: A random sample from the given array with a given probability.
    """
    return arr[np.searchsorted(np.cumsum(prob), np.random.random(), side="right")]

@jit(nopython=True)
def rand_choice_nb(arr, size=None):
    return np.random.choice(arr, size=size)

@jit(nopython=True)
def arange_nb(stop, start=0):
    return np.arange(start, stop)

@jit(nopython=True)
def generate_mixup_nb(batch_size, batch_obs, batch_actions, mixup_obs, mixup_actions):
    lmda = np.random.random(size=(batch_size, 1))
    obs = lmda * batch_obs + (1 - lmda) * mixup_obs
    actions = lmda * batch_actions + (1 - lmda) * mixup_actions
    return obs, actions

