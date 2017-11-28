import argparse
import random

from railrl.envs.multitask.reacher_7dof import Reacher7DofXyzGoalState
from railrl.state_distance.vectorized_sdql import (
    VectorizedTauSdql,
    VectorizedDeltaTauSdql,
)
from torch.nn import functional as F

import railrl.misc.hyperparameter as hyp
import railrl.torch.pytorch_util as ptu
from railrl.data_management.her_replay_buffer import HerReplayBuffer
from railrl.data_management.split_buffer import SplitReplayBuffer
from railrl.envs.multitask.reacher_env import (
    GoalStateSimpleStateReacherEnv)
from railrl.envs.wrappers import convert_gym_space, normalize_box
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import run_experiment
from railrl.misc.ml_util import IntRampUpSchedule, \
    ConstantSchedule
from railrl.policies.state_distance import \
    UnconstrainedOcWithGoalConditionedModel, UnconstrainedOcWithImplicitModel
from railrl.state_distance.exploration import \
    UniversalPolicyWrappedWithExplorationStrategy
from railrl.state_distance.networks import (
    FFUniversalPolicy,
    VectorizedGoalStructuredUniversalQfunction,
    GoalStructuredUniversalQfunction, GoalConditionedDeltaModel,
    TauBinaryGoalConditionedDeltaModel)
from railrl.state_distance.state_distance_q_learning import \
    HorizonFedStateDistanceQLearning
from railrl.torch.modules import HuberLoss


def experiment(variant):
    env_class = variant['env_class']
    env = env_class(**variant['env_params'])
    env = normalize_box(
        env,
        **variant['normalize_params']
    )

    observation_space = convert_gym_space(env.observation_space)
    action_space = convert_gym_space(env.action_space)
    qf = variant['qf_class'](
        int(observation_space.flat_dim),
        int(action_space.flat_dim),
        env.goal_dim,
        **variant['qf_params']
    )
    policy = FFUniversalPolicy(
        int(observation_space.flat_dim),
        int(action_space.flat_dim),
        env.goal_dim,
        **variant['policy_params']
    )
    epoch_discount_schedule = None
    epoch_discount_schedule_class = variant['epoch_discount_schedule_class']
    if epoch_discount_schedule_class is not None:
        epoch_discount_schedule = epoch_discount_schedule_class(
            **variant['epoch_discount_schedule_params']
        )
    qf_criterion = variant['qf_criterion_class'](
        **variant['qf_criterion_params']
    )
    es = variant['sampler_es_class'](
        action_space=action_space,
        **variant['sampler_es_params']
    )
    raw_explore_policy = variant['raw_explore_policy']
    if isinstance(qf, VectorizedGoalStructuredUniversalQfunction):
        oc_policy = UnconstrainedOcWithImplicitModel(
            qf,
            env,
            policy,
            **variant['oc_policy_params']
        )
    else:
        oc_policy = UnconstrainedOcWithGoalConditionedModel(
            qf,
            env,
            policy,
            **variant['oc_policy_params']
        )
    if raw_explore_policy == 'ddpg':
        raw_exploration_policy = policy
    else:
        raw_exploration_policy = oc_policy
    exploration_policy = UniversalPolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=raw_exploration_policy,
    )
    if variant['algo_params']['sample_train_goals_from'] == 'her':
        replay_buffer = SplitReplayBuffer(
            HerReplayBuffer(
                env=env,
                **variant['her_replay_buffer_params']
            ),
            HerReplayBuffer(
                env=env,
                **variant['her_replay_buffer_params']
            ),
            fraction_paths_in_train=.5,
        )
    else:
        replay_buffer = None
    if variant['eval_with_oc_policy']:
        eval_policy = oc_policy
    else:
        eval_policy = policy
    algo = variant['algo_class'](
        env,
        qf,
        policy,
        exploration_policy,
        epoch_discount_schedule=epoch_discount_schedule,
        qf_criterion=qf_criterion,
        replay_buffer=replay_buffer,
        eval_policy=eval_policy,
        **variant['algo_params']
    )
    if ptu.gpu_enabled():
        algo.cuda()
    algo.train()


algo_class_to_qf_class = {
    VectorizedTauSdql: VectorizedGoalStructuredUniversalQfunction,
    VectorizedDeltaTauSdql: GoalConditionedDeltaModel,
    # VectorizedDeltaTauSdql: TauBinaryGoalConditionedDeltaModel,
    HorizonFedStateDistanceQLearning: GoalStructuredUniversalQfunction,
}


def complete_variant(variant):
    algo_class = variant['algo_class']
    variant['algo_params']['sparse_reward'] = not (
        algo_class == VectorizedDeltaTauSdql
    )
    if 'qf_class' not in variant:
        variant['qf_class'] = algo_class_to_qf_class[algo_class]
    if variant['epoch_discount_schedule_class'] == ConstantSchedule:
        discount = variant['epoch_discount_schedule_params']['value']
        variant['algo_params']['discount'] = discount
        if variant['qf_class'] == TauBinaryGoalConditionedDeltaModel:
            variant['qf_params']['max_tau'] = discount
    return variant


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--replay_path', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    n_seeds = 1
    mode = "local"
    exp_prefix = "dev-vectorized-train-q"
    run_mode = "none"
    snapshot_mode = "last"

    # n_seeds = 1
    # mode = "ec2"
    exp_prefix = "tdm-reacher-6dof-xyz-test"
    # run_mode = 'grid'
    # snapshot_mode = "gap_and_last"

    version = "na"
    num_configurations = 50  # for random mode
    snapshot_gap = 50
    use_gpu = True
    if mode != "local":
        use_gpu = False

    # noinspection PyTypeChecker
    algo_class = VectorizedTauSdql
    # algo_class = VectorizedDeltaTauSdql
    qf_class = algo_class_to_qf_class[algo_class]

    # env_class = Reacher7DofAngleGoalState
    # env_class = GoalCosSinStateXYAndCosSinReacher2D
    env_class = Reacher7DofXyzGoalState
    # env_class = JointOnlyPusherEnv
    # env_class = GoalStateSimpleStateReacherEnv
    # env_class = GoalXYStateXYAndCosSinReacher2D
    # env_class = Reacher7DofFullGoalState
    # env_class = Reacher7DofGoalStateEverything
    # env_class = HandXYPusher2DEnv
    # env_class = HandCylinderXYPusher2DEnv
    # env_class = LessShapeHandCylinderXYPusher2DEnv
    # env_class = HandPuckXYZPusher3DEnv
    # env_class = GoalXVelHalfCheetah
    # env_class = GoalXYVelAnt
    # env_class = FullStatePusher2DEnv
    # env_class = FixedHandXYPusher2DEnv
    # env_class = CylinderXYPusher2DEnv
    max_path_length = 100
    max_tau = 10
    num_epochs = 101
    replay_buffer_size = 200000
    variant = dict(
        version=version,
        algo_params=dict(
            num_epochs=num_epochs,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            num_updates_per_env_step=25,
            use_soft_update=True,
            tau=0.001,
            batch_size=1,
            discount=max_tau,
            qf_learning_rate=1e-3,
            policy_learning_rate=1e-4,
            sample_rollout_goals_from='environment',
            sample_train_goals_from='her',
            qf_weight_decay=0.,
            max_path_length=max_path_length,
            replay_buffer_size=replay_buffer_size,
            prob_goal_is_next_state=0,
            termination_threshold=0,
            render=args.render,
            save_replay_buffer=True,
            cycle_taus_for_rollout=True,
            sl_grad_weight=1,
            num_sl_batches_per_rl_batch=0,
            only_do_sl=False,
            # cycle_taus_for_rollout=False,
            # num_sl_batches_per_rl_batch=1,
        ),
        # eval_with_oc_policy=True,
        eval_with_oc_policy=False,
        her_replay_buffer_params=dict(
            max_size=replay_buffer_size,
            num_goals_to_sample=4,
            # goal_sample_strategy='store',
        ),
        raw_explore_policy='ddpg',
        oc_policy_params=dict(
            sample_size=10000,
        ),
        qf_params=dict(
            hidden_sizes=[300, 300],
            hidden_activation=F.relu,
            # max_tau=max_tau,
        ),
        policy_params=dict(
            fc1_size=300,
            fc2_size=300,
        ),
        epoch_discount_schedule_class=IntRampUpSchedule,
        epoch_discount_schedule_params=dict(
            min_value=10,
            max_value=10,
            ramp_duration=num_epochs,
        ),
        algo_class=algo_class,
        env_class=env_class,
        env_params=dict(),
        normalize_params=dict(
            # obs_mean=None,
            # obs_std=[1, 1, 1, 1, 20, 20],
        ),
        sampler_es_class=OUStrategy,
        # sampler_es_class=GaussianStrategy,
        sampler_es_params=dict(
            theta=0.1,
            max_sigma=0.1,
            min_sigma=0.1,
        ),
        qf_criterion_class=HuberLoss,
        qf_criterion_params=dict(),
        exp_prefix=exp_prefix,
    )
    if run_mode == 'grid':
        search_space = {
            # 'raw_explore_policy': [
            #     'oc',
            #     'ddpg',
            # ],
            # 'oc_policy_params.reward_function': [
            #     FullStatePusher2DEnv_move_hand_to_target_position_oc_reward_on_goals,
            #     FullStatePusher2DEnv_move_joints_to_target_joint,
            # ],
            # 'algo_class': [
            #     HorizonFedStateDistanceQLearning,
            #     VectorizedDeltaTauSdql,
            #     VectorizedTauSdql,
            # ],
            # 'qf_params.hidden_sizes': [
            #     [64, 64],
            #     [300, 300],
            #     [100, 100, 100],
            # ],
            # 'qf_class': [
                # GoalConditionedDeltaModel,
                # TauBinaryGoalConditionedDeltaModel,
            # ],
            # 'algo_params.goal_dim_weights': [
            #     (.1, .1),
            #     (1, .1),
            #     (10, 10),
            #     (.01, .01, 1, 1),
            #     (.1, .1, 1, 1),
            #     (0, 0, 1, 1),
            #     (1, 1, 1, 1),
            #     (.1, .1, .1, .1),
            #     (10, 10, 10, 10),
            #     (1, 1, 0.1, 0.1),
            #     (1, 1, 0.01, 0.01),
            #     (1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
            #     (5, 5, 5, 5, 5, 5, 5, 5, 5, 5),
            #     (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 5, 5, 1, 1),
            #     (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1, 1, 1, 1),
            #     (0, 0, 0, 0, 0, 0, 1, 1, 1, 1),
            #     (0, 0, 0, 0, 0, 0, 1, 1, .1, .1),
            # ],
            # 'env_class': [
                # GoalStateSimpleStateReacherEnv,
                # FullStatePusher2DEnv,
                # NoShapeHandCylinderXYPusher2DEnv,
                # HandCylinderXYPusher2DEnv,
                # Reacher7DofFullGoalState,
                # Reacher7DofGoalStateEverything,
                # JointOnlyPusherEnv,
                # MultitaskPusher2DEnv,
                # CylinderXYPusher2DEnv,
            # ],
            # 'epoch_discount_schedule_params.max_value': [
            #     1,
            #     5,
            #     10,
            #     15,
            #     20,
            # ],
            # 'algo_params.use_soft_update': [
            #     True,
            # ],
            # 'algo_params.target_hard_update_period': [
            #     1,
            #     100,
            #     1000,
            # ],
            # 'eval_with_oc_policy': [
            #     False,
            #     True,
            # ],
            # 'algo_params.tau': [
            #     1e-1, 1e-2, 1e-3, 1e-4
            # ],
            # 'algo_params.sl_grad_weight': [
            #     0.01,
            #     0.1,
            #     1,
            #     10,
            # ],
            # 'her_replay_buffer_params'
            # '.fraction_goals_are_rollout_goals': [
            #     None,
            #     0,
            #     0.5,
            #     1,
            # ]
            # 'her_replay_buffer_params.goal_sample_strategy': [
            #     'online',
            #     'store',
            # ],
        }
        sweeper = hyp.DeterministicHyperparameterSweeper(
            search_space, default_parameters=variant,
        )
        for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
            for i in range(n_seeds):
                seed = random.randint(0, 10000)
                variant = complete_variant(variant)
                run_experiment(
                    experiment,
                    exp_prefix=exp_prefix,
                    seed=seed,
                    mode=mode,
                    variant=variant,
                    exp_id=exp_id,
                    use_gpu=use_gpu,
                    snapshot_mode=snapshot_mode,
                    snapshot_gap=snapshot_gap,
                )
    else:
        for _ in range(n_seeds):
            seed = random.randint(0, 10000)
            variant = complete_variant(variant)
            run_experiment(
                experiment,
                exp_prefix=exp_prefix,
                seed=seed,
                mode=mode,
                variant=variant,
                exp_id=0,
                use_gpu=use_gpu,
                snapshot_mode=snapshot_mode,
                snapshot_gap=snapshot_gap,
            )
