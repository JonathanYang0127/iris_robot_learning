import argparse

from torch.nn import functional as F

import railrl.torch.pytorch_util as ptu
from railrl.algos.state_distance.state_distance_q_learning import (
    StateDistanceQLearning,
    HorizonFedStateDistanceQLearning,
)
from railrl.envs.multitask.baxter_env import MultiTaskBaxterEnv
from railrl.envs.wrappers import convert_gym_space
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import run_experiment
from railrl.misc.ml_util import ConstantSchedule
from railrl.networks.state_distance import (
    FFUniversalPolicy,
    FlatUniversalQfunction,
)
from railrl.torch.modules import HuberLoss
from railrl.torch.state_distance.exploration import \
    UniversalPolicyWrappedWithExplorationStrategy


def experiment(variant):
    env_params=variant['env_params']
    env = MultiTaskBaxterEnv(**env_params)

    observation_space = convert_gym_space(env.observation_space)
    action_space = convert_gym_space(env.action_space)
    qf = FlatUniversalQfunction(
        int(observation_space.flat_dim),
        int(action_space.flat_dim),
        env.goal_dim,
        **variant['qf_params'],
    )

    policy = FFUniversalPolicy(
        int(observation_space.flat_dim),
        int(action_space.flat_dim),
        env.goal_dim,
        **variant['policy_params']
    )

    es = variant['sampler_es_class'](
        action_space=action_space,
        **variant['sampler_es_params']
    )

    exploration_policy = UniversalPolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    epoch_discount_schedule = variant['epoch_discount_schedule_class'](
        **variant['epoch_discount_schedule_params']
    )

    algo = HorizonFedStateDistanceQLearning(
        env,
        qf,
        policy,
        exploration_policy,
        qf_criterion=HuberLoss(),
        epoch_discount_schedule=epoch_discount_schedule,
        **variant['algo_params']
    )
    if ptu.gpu_enabled():
        algo.cuda()
    algo.train()

experiments=[
    'joint_angle|fixed_angle',
    'joint_angle|varying_angle',
    'end_effector_position|fixed_ee',
    'end_effector_position|varying_ee',
    'end_effector_position_orientation|fixed_ee',
    'end_effector_position_orientation|varying_ee'
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    n_seeds = 1
    use_gpu = True
    max_path_length = 300
    variant = dict(
        algo_params=dict(
            num_epochs=101,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            num_updates_per_env_step=1,
            use_soft_update=True,
            tau=0.001,
            batch_size=500,
            discount=0.99,
            sample_goals_from='environment',
            sample_discount=False,
            qf_weight_decay=0.,
            max_path_length=max_path_length,
            use_new_data=True,
            replay_buffer_size=1000000,
            prob_goal_state_is_next_state=0,
            termination_threshold=0,
            render=args.render,
            save_replay_buffer=True,
            sparse_reward=False,
        ),
        algo_class=HorizonFedStateDistanceQLearning,
        qf_params=dict(
            hidden_sizes=[400, 300],
            hidden_activation=F.softplus,
        ),
        policy_params=dict(
            fc1_size=400,
            fc2_size=300,
        ),
        sampler_es_class=OUStrategy,
        sampler_es_params=dict(
            theta=0.15,
            max_sigma=0.2,
            min_sigma=0.2,
        ),
        env_params=dict(
            arm_name='left',
            safety_box=False,
            loss='huber',
            huber_delta=10,
            experiment=experiments[2],
            reward_magnitude=10,
        ),
        epoch_discount_schedule_class=ConstantSchedule,
        epoch_discount_schedule_params=dict(
            value=5,
        ),
    )
    run_experiment(
        experiment,
        exp_prefix="sdql-example",
        mode="local",
        variant=variant,
        exp_id=0,
        use_gpu=use_gpu,
        snapshot_mode="last",
    )