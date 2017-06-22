"""
Exampling of running DDPG on HalfCheetah.
"""
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import run_experiment
from railrl.policies.torch import FeedForwardPolicy
from railrl.qfunctions.torch import FeedForwardQFunction
from railrl.torch.ddpg import DDPG
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.envs.mujoco.hopper_env import HopperEnv

from rllab.envs.normalized_env import normalize


def example(variant):
    env = HalfCheetahEnv()
    # env = PointEnv()
    # env = gym_env("Pendulum-v0")
    # env = HopperEnv()
    env = normalize(env)
    es = OUStrategy(action_space=env.action_space)
    qf = FeedForwardQFunction(
        int(env.observation_space.flat_dim),
        int(env.action_space.flat_dim),
        400,
        300,
    )
    policy = FeedForwardPolicy(
        int(env.observation_space.flat_dim),
        int(env.action_space.flat_dim),
        400,
        300,
    )
    algorithm = DDPG(
        env,
        exploration_strategy=es,
        qf=qf,
        policy=policy,
        **variant['algo_params']
    )
    algorithm.train()


if __name__ == "__main__":
    variant = dict(
        algo_params=dict(
            num_epochs=50,
            num_steps_per_epoch=10000,
            num_steps_per_eval=1000,
            target_hard_update_period=10000,
            # use_soft_update=True,
            batch_size=128,
            max_path_length=1000,
        )
    )
    run_experiment(
        example,
        exp_prefix="6-21-pytorch-ddpg-no-bn-cheetah",
        seed=0,
        mode='here',
        variant=variant,
        use_gpu=True,
    )
