from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.torch_rl_algorithm import TorchTrainer


def filter_dict(key_prefix, batch_dict):
    assert isinstance(key_prefix, str)
    return dict([(key, batch_dict[key]) for key in batch_dict.keys()
                 if key.find(key_prefix) == 0])


class RealNVPTrainer(TorchTrainer):
    def __init__(
            self,
            env,
            bijector,
            lr=1e-4,
            optimizer_class=optim.Adam,
            render_eval_paths=False,
            clip_gradients_by_norm=False,
            clip_gradients_by_norm_threshold=50.0
    ):
        super().__init__()
        self.env = env
        self.bijector = bijector

        self.render_eval_paths = render_eval_paths

        self.optimizer = optimizer_class(
            self.bijector.parameters(),
            lr=lr,
        )
        self.clip_gradients_by_norm = clip_gradients_by_norm
        self.clip_gradients_by_norm_threshold = clip_gradients_by_norm_threshold

        self._current_epoch = 0

        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

    def compute_loss(self, batch):
        terminals = batch['terminals']
        actions = batch['actions']
        obs = batch['observations']
        next_obs = batch['next_observations']

        # import IPython; IPython.embed()
        out_forwardpass = self.bijector.log_prob_chain(actions, obs)
        loss = - torch.mean(out_forwardpass)
        return loss

    def train_from_torch(self, batch, online=False):
        self._current_epoch += 1

        """
        Optimize exact log-likelihood of observed actions. 
        """

        loss = self.compute_loss(batch)
        self.optimizer.zero_grad()
        loss.backward()
        if self.clip_gradients_by_norm:
            torch.nn.utils.clip_grad_norm_(
                self.bijector.parameters(),
                self.clip_gradients_by_norm_threshold
            )
        self.optimizer.step()

        """
        Some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            total_norm = 0.0
            for p in self.bijector.parameters():
                param_norm = ptu.get_numpy(p.grad.data.norm(2))
                total_norm += param_norm ** 2
            total_norm = total_norm ** (1. / 2)
            self.eval_statistics['gradient_norm'] = np.mean(total_norm)
            self.eval_statistics['loss'] = np.mean(ptu.get_numpy(loss))

        self._n_train_steps_total += 1

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.bijector
        ]

    def get_snapshot(self):
        return dict(
            bijector=self.bijector,
        )