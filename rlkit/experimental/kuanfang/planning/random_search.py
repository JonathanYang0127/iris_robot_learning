import numpy as np
import torch

from rlkit.torch import pytorch_util as ptu


def _compute_dist(h_t, h_g):
    h_t = torch.flatten(h_t, start_dim=-3)
    h_g = torch.flatten(h_g, start_dim=-3)
    return torch.sum((h_g - h_t) ** 2, dim=-1)


class RandomSearch(object):

    def __init__(
            self,
            model,

            num_samples=1024,

            progress_thresh=2.0,
    ):

        self.vqvae = model['vqvae']
        self.affordance = model['affordance']

        self.num_samples = num_samples

        self.progress_thresh = progress_thresh

    def __call__(self, s_0, s_g, max_steps=5):
        # Pre-process the input data.
        s_0 = ptu.from_numpy(s_0[np.newaxis, :, :, :])
        h_0 = self.vqvae.encode(s_0, flatten=False)

        s_g = ptu.from_numpy(s_g[np.newaxis, :, :, :])
        h_g = self.vqvae.encode(s_g, flatten=False)

        # Tile the sequence.
        h_0 = h_0.repeat((self.num_samples, 1, 1, 1))
        h_g = h_g.repeat((self.num_samples, 1, 1, 1))

        # Recursive prediction.
        h_preds = []
        h_t = h_0
        for t in range(max_steps):
            z_t = self.affordance.sample_prior(h_t.shape[0])
            z_t = ptu.from_numpy(z_t)
            h_pred = self.affordance.decode(z_t, cond=h_t)

            h_preds.append(h_pred)
            h_t = h_pred

        # Rank the plans.
        h_preds = torch.stack(h_preds, 1)
        dists = _compute_dist(
            h_preds,
            torch.stack([h_g] * max_steps, 1),
        )
        min_dists, min_steps = torch.min(dists, 1)

        print(min_dists.shape, min_steps.shape)

        top_dist, top_ind = torch.min(min_dists, 0)
        top_step = min_steps[top_ind]

        print('top_ind: ', top_ind,
              'top_step: ', top_step,
              'top_dist: ', top_dist)

        top_h_pred = h_preds[top_ind, :]
        top_s_pred = self.vqvae.decode(top_h_pred)

        print('top_s_pred: ', top_s_pred.shape)

        # Optional: Prevent the random actions after achieving the goal.
        top_s_pred[top_step:] = top_s_pred[top_step:top_step + 1]

        # Post-processing the plan.
        top_s_pred = top_s_pred[:, None, ...]
        sequence = torch.unbind(top_s_pred, 0)
        sequence = [s_0] + list(sequence)

        return sequence
