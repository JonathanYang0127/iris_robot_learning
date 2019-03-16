"""
Run DQN on grid world.
"""

import gym
import numpy as np
from torch import nn as nn

import railrl.torch.pytorch_util as ptu
from railrl.launchers.launcher_util import setup_logger
from railrl.torch.dqn.dqn import DQN
from railrl.torch.networks import Mlp


def experiment(variant):
    env = gym.make('CartPole-v0')
    training_env = gym.make('CartPole-v0')

    qf = Mlp(
        hidden_sizes=[32, 32],
        input_size=int(np.prod(env.observation_space.shape)),
        output_size=env.action_space.n,
    )
    target_qf = Mlp(
        hidden_sizes=[32, 32],
        input_size=int(np.prod(env.observation_space.shape)),
        output_size=env.action_space.n,
    )
    qf_criterion = nn.MSELoss()
    # Use this to switch to DoubleDQN
    # algorithm = DoubleDQN(
    algorithm = DQN(
        env,
        training_env=training_env,
        qf=qf,
        target_qf=target_qf,
        qf_criterion=qf_criterion,
        **variant['algo_params']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algo_params=dict(
            num_epochs=500,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            batch_size=128,
            max_path_length=200,
            discount=0.99,
            epsilon=0.2,
            tau=0.001,
            hard_update_period=1000,
            save_environment=False,  # Can't serialize CartPole for some reason
        ),
    )
    setup_logger('name-of-experiment', variant=variant)
    experiment(variant)
