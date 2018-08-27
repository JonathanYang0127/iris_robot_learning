"""
Try increasing gamma slowly to see how it affects DDPG.
"""
import random

from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import run_experiment
from railrl.misc.ml_util import LinearSchedule
from railrl.torch.networks import FeedForwardQFunction, FeedForwardPolicy
from railrl.torch.ddpg import DDPG
import railrl.torch.pytorch_util as ptu

from rllab.envs.mujoco.ant_env import AntEnv
from rllab.envs.mujoco.hopper_env import HopperEnv
from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.envs.normalized_env import normalize


def experiment(variant):
    env = variant['env_class']()
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
    epoch_discount_schedule_class = variant['epoch_discount_schedule_class']
    epoch_discount_schedule = epoch_discount_schedule_class(
        **variant['epoch_discount_schedule_params']
    )
    algorithm = DDPG(
        env,
        exploration_strategy=es,
        qf=qf,
        policy=policy,
        epoch_discount_schedule=epoch_discount_schedule,
        **variant['algo_params']
    )
    if ptu.gpu_enabled():
        algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algo_params=dict(
            num_epochs=100,
            num_steps_per_epoch=10000,
            num_steps_per_eval=1000,
            use_soft_update=True,
            tau=1e-2,
            batch_size=128,
            max_path_length=1000,
            discount=0.99,
            qf_learning_rate=1e-3,
            policy_learning_rate=1e-4,
        ),
        version="DDPG",
        epoch_discount_schedule_class=LinearSchedule,
        epoch_discount_schedule_params=dict(
            min_value=0.,
            max_value=0.99,
            ramp_duration=99,
        ),
    )
    for env_class in [
        SwimmerEnv,
        HalfCheetahEnv,
        AntEnv,
        HopperEnv,
    ]:
        variant['env_class'] = env_class
        variant['version'] = str(env_class)
        for _ in range(5):
            seed = random.randint(0, 999999)
            run_experiment(
                experiment,
                exp_prefix="ddpg-increase-gamma",
                seed=seed,
                mode='ec2',
                variant=variant,
                use_gpu=False,
            )
