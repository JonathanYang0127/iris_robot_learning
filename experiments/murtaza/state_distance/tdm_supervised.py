from railrl.data_management.her_replay_buffer import HerReplayBuffer
from railrl.envs.multitask.point2d import MultitaskPoint2DEnv
from railrl.envs.multitask.reacher_7dof import Reacher7DofXyzGoalState
from railrl.envs.wrappers import NormalizedBoxEnv
from railrl.exploration_strategies.base import (
    PolicyWrappedWithExplorationStrategy
)
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.state_distance.tdm_supervised import TdmSupervised
import railrl.torch.pytorch_util as ptu
from railrl.state_distance.tdm_networks import TdmPolicy
import railrl.misc.hyperparameter as hyp
from railrl.launchers.launcher_util import run_experiment
import numpy as np

def experiment(variant):
    env = NormalizedBoxEnv(Reacher7DofXyzGoalState()) #try full state reacher
    es = OUStrategy(action_space=env.action_space)
    policy = TdmPolicy(
        env=env,
        **variant['policy_kwargs']
    )
    replay_buffer_size = variant['algo_params']['base_kwargs']['replay_buffer_size']
    replay_buffer = HerReplayBuffer(replay_buffer_size, env)
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    algorithm = TdmSupervised(
        env,
        replay_buffer=replay_buffer,
        policy=policy,
        exploration_policy=exploration_policy,
        **variant['algo_params']
    )
    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        policy_kwargs=dict(
            hidden_sizes=[300, 300],
        ),
        algo_params=dict(
            base_kwargs=dict(
                num_epochs=100,
                num_steps_per_epoch=1000,
                num_steps_per_eval=1000,
                max_path_length=100,
                num_updates_per_env_step=1,
                batch_size=128,
                discount=1,
                reward_scale=100,
                replay_buffer_size=1000000,
                render=True,
            ),
            tdm_kwargs=dict(
                max_tau=10,
            ),
        ),
    )
    search_space = {
        'algo_params.base_kwargs.reward_scale': [
            1,
            # 10,
            # 100,
            # 1000,
            # 10000,
        ],
        'algo_params.tdm_kwargs.max_tau': [
            0,
            # 10,
            # 15,
            # 20,
        ],
        'algo_params.policy_criterion':[
            'MSE',
            # 'Huber',
        ]
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        run_experiment(
            experiment,
            seed=np.random.randint(1, 10004),
            variant=variant,
            exp_id=exp_id,
            exp_prefix='test',
            mode='local',
        )
