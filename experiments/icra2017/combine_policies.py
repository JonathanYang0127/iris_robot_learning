import random

import joblib
import numpy as np

from railrl.envs.mujoco.pusher3dof import PusherEnv3DOF, get_snapshots_and_goal
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import run_experiment
from railrl.policies.torch import FeedForwardPolicy
from railrl.qfunctions.torch import FeedForwardQFunction
from railrl.torch.algos.combine_ddpg_qfs import DdpgQfCombiner
from railrl.torch.ddpg import DDPG
import railrl.misc.hyperparameter as hyp
import railrl.torch.pytorch_util as ptu

from rllab.envs.normalized_env import normalize


def experiment(variant):
    env = PusherEnv3DOF(**variant['env_params'])
    env = normalize(env)
    ddpg1_snapshot_dict = joblib.load(variant['ddpg1_snapshot_path'])
    ddpg2_snapshot_dict = joblib.load(variant['ddpg2_snapshot_path'])
    policy = FeedForwardPolicy(
        int(env.observation_space.flat_dim),
        int(env.action_space.flat_dim),
        100,
        100,
    )
    algorithm = DdpgQfCombiner(
        env=env,
        qf1=ddpg1_snapshot_dict['qf'],
        qf2=ddpg2_snapshot_dict['qf'],
        policy=policy,
        replay_buffer1=ddpg1_snapshot_dict['replay_buffer'],
        replay_buffer2=ddpg2_snapshot_dict['replay_buffer'],
        **variant['algo_params']
    )
    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()


if __name__ == '__main__':
    num_configurations = 1  # for random mode
    n_seeds = 1
    mode = "here"
    exp_prefix = "dev-combine-policies"
    version = "Dev"
    run_mode = "none"

    # n_seeds = 10
    # mode = "ec2"
    exp_prefix = "combine-policies"
    # version = "Dev"
    # run_mode = 'grid'

    use_gpu = True
    if mode != "here":
        use_gpu = False

    vertical_pos = 'left'
    horizontal_pos = 'bottom'

    exp_prefix += "--{}-{}".format(vertical_pos, horizontal_pos)
    ddpg1_snapshot_path, ddpg2_snapshot_path, x_goal, y_goal = (
        get_snapshots_and_goal(
            vertical_pos=vertical_pos,
            horizontal_pos=horizontal_pos,
        )
    )
    variant = dict(
        version=version,
        ddpg1_snapshot_path=ddpg1_snapshot_path,
        ddpg2_snapshot_path=ddpg2_snapshot_path,
        algo_params=dict(
            num_epochs=100,
            num_steps_per_epoch=1000,
            policy_learning_rate=1e-3,
            batch_size=128,
            num_steps_per_eval=900,
            max_path_length=300,
            discount=0.99
        ),
        env_params=dict(
            goal=(x_goal, y_goal),
        ),
    )
    if run_mode == 'grid':
        search_space = {
            'algo_params.policy_learning_rate': [1e-3, 1e-4, 1e-5],
        }
        sweeper = hyp.DeterministicHyperparameterSweeper(
            search_space, default_parameters=variant,
        )
        for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
            for i in range(n_seeds):
                seed = random.randint(0, 10000)
                run_experiment(
                    experiment,
                    exp_prefix=exp_prefix,
                    seed=seed,
                    mode=mode,
                    variant=variant,
                    exp_id=exp_id,
                    use_gpu=use_gpu,
                    sync_s3_log=True,
                    sync_s3_pkl=True,
                    periodic_sync_interval=600,
                )
    else:
        for _ in range(n_seeds):
            seed = random.randint(0, 10000)
            run_experiment(
                experiment,
                exp_prefix=exp_prefix,
                seed=seed,
                mode=mode,
                variant=variant,
                exp_id=0,
                use_gpu=use_gpu,
                sync_s3_log=True,
                sync_s3_pkl=True,
                periodic_sync_interval=600,
            )
