import random

import gym
import numpy as np

import railrl.torch.pytorch_util as ptu
from railrl.launchers.launcher_util import run_experiment
from railrl.sac.policies import TanhGaussianPolicy
from railrl.sac.sac import SoftActorCritic
from railrl.torch.networks import FlattenMlp
import sys
sys.path.append('/home/murtaza/Documents/objectattention/')
from singleobj_visreward import SingleObjVisRewardEnv

env = SingleObjVisRewardEnv()

def experiment(variant):
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    net_size = variant['net_size']
    qf = FlattenMlp(
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim + action_dim,
        output_size=1,
    )
    vf = FlattenMlp(
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim,
        output_size=1,
    )
    policy = TanhGaussianPolicy(
        hidden_sizes=[net_size, net_size],
        obs_dim=obs_dim,
        action_dim=action_dim,
    )
    algorithm = SoftActorCritic(
        env=env,
        policy=policy,
        qf=qf,
        vf=vf,
        **variant['algo_params']
    )
    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algo_params=dict(
            num_epochs=100,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            batch_size=64,
            max_path_length=100,
            discount=0.99,
            reward_scale=3,
            soft_target_tau=0.001,
            policy_lr=3E-4,
            qf_lr=3E-4,
            vf_lr=3E-4,
        ),
        net_size=100,
    )
    seed = random.randint(0, 10000)
    exp_prefix = 'singleobj_visreward_SAC'
    mode='local'
    run_experiment(
        experiment,
        seed=seed,
        variant=variant,
        exp_prefix=exp_prefix,
        mode=mode,
    )
