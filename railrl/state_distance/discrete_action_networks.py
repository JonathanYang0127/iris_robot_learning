import torch
import numpy as np
import railrl.torch.pytorch_util as ptu
from railrl.policies.base import SerializablePolicy
from railrl.torch.core import PyTorchModule
from railrl.torch.networks import Mlp


class VectorizedDiscreteQFunction(Mlp):
    """
    If `states is size BATCH_SIZE x OBS_DIM,
    Q(states, taus) outputs a tensor of shape
        BATCH_SIZE x ACTION_DIM x GOAL_DIM
    """
    def __init__(
            self,
            hidden_sizes,
            action_dim,
            observation_dim,
            goal_dim,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(
            hidden_sizes=hidden_sizes,
            output_size=action_dim * goal_dim,
            input_size=observation_dim + goal_dim + 1,
            **kwargs
        )
        self.observation_dim = observation_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim

    def forward(self, *inputs):
        h = torch.cat(inputs, dim=1)
        batch_size = h.size()[0]
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        return self.output_activation(self.last_fc(h)).view(
            batch_size, self.action_dim, self.goal_dim
        )


class ArgmaxDiscreteTdmPolicy(PyTorchModule, SerializablePolicy):
    def __init__(self, qf, goal_dim_weights=None):
        self.save_init_params(locals())
        super().__init__()
        self.qf = qf
        if goal_dim_weights is not None:
            goal_dim_weights = np.expand_dims(np.array(goal_dim_weights), 0)
        self.goal_dim_weights = goal_dim_weights

    def get_action(self, obs):
        obs = np.expand_dims(obs, axis=0)
        obs = ptu.np_to_var(obs, requires_grad=False).float()
        q_values = self.qf(obs).squeeze(0)
        # Take the action that has the best sum across all weights
        q_values_np = ptu.get_numpy(q_values)
        if self.goal_dim_weights is not None:
            q_values_np = q_values_np * self.goal_dim_weights
        q_values_np = q_values_np.sum(axis=1)
        return q_values_np.argmax(), {}
