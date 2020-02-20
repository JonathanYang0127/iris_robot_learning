"""
This should results in an average return of ~3000 by the end of training.

Usually hits 3000 around epoch 80-100. Within a see, the performance will be
a bit noisy from one epoch to the next (occasionally dips dow to ~2000).

Note that one epoch = 5k steps, so 200 epochs = 1 million steps.
"""
import gym

import railrl.torch.pytorch_util as ptu
from railrl.data_management.obs_dict_replay_buffer import ObsDictRelabelingBuffer
from railrl.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from railrl.exploration_strategies.gaussian_and_epislon import \
    GaussianAndEpislonStrategy
from railrl.launchers.launcher_util import setup_logger
from railrl.samplers.data_collector import GoalConditionedPathCollector
from railrl.torch.her.her import HERTrainer
from railrl.torch.networks import FlattenMlp, TanhMlpPolicy
from railrl.torch.td3.td3 import TD3 as TD3Trainer
from railrl.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

from railrl.launchers.launcher_util import run_experiment
import railrl.misc.hyperparameter as hyp

from railrl.launchers.exp_launcher import rl_experiment

def experiment(variant):
    from multiworld.envs.mujoco import register_mujoco_envs
    register_mujoco_envs()
    eval_env = gym.make('SawyerReachXYZEnv-v0')
    expl_env = gym.make('SawyerReachXYZEnv-v0')
    observation_key = 'state_observation'
    desired_goal_key = 'state_desired_goal'
    achieved_goal_key = desired_goal_key.replace("desired", "achieved")
    es = GaussianAndEpislonStrategy(
        action_space=expl_env.action_space,
        max_sigma=.2,
        min_sigma=.2,  # constant sigma
        epsilon=.3,
    )
    obs_dim = expl_env.observation_space.spaces['observation'].low.size
    goal_dim = expl_env.observation_space.spaces['desired_goal'].low.size
    action_dim = expl_env.action_space.low.size
    qf1 = FlattenMlp(
        input_size=obs_dim + goal_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + goal_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    target_qf1 = FlattenMlp(
        input_size=obs_dim + goal_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    target_qf2 = FlattenMlp(
        input_size=obs_dim + goal_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    policy = TanhMlpPolicy(
        input_size=obs_dim + goal_dim,
        output_size=action_dim,
        **variant['policy_kwargs']
    )
    target_policy = TanhMlpPolicy(
        input_size=obs_dim + goal_dim,
        output_size=action_dim,
        **variant['policy_kwargs']
    )
    expl_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    replay_buffer = ObsDictRelabelingBuffer(
        env=eval_env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **variant['replay_buffer_kwargs']
    )
    trainer = TD3Trainer(
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        target_policy=target_policy,
        **variant['trainer_kwargs']
    )
    trainer = HERTrainer(trainer)
    eval_path_collector = GoalConditionedPathCollector(
        eval_env,
        policy,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
    )
    expl_path_collector = GoalConditionedPathCollector(
        expl_env,
        expl_policy,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algo_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    variant = dict(
        env_id='SawyerReachXYZEnv-v0',
        rl_variant=dict(
            do_state_exp=True,
            algo_kwargs=dict(
                num_epochs=300,
                batch_size=128,
                num_eval_steps_per_epoch=1000,
                num_expl_steps_per_train_loop=1000,
                num_trains_per_train_loop=1000,
            ),
            max_path_length=100,
            td3_trainer_kwargs=dict(),
            twin_sac_trainer_kwargs=dict(),
            replay_buffer_kwargs=dict(
                max_size=int(1E6),
                fraction_goals_rollout_goals=0.2,
                fraction_goals_env_goals=0.5,
            ),
            exploration_noise=0.1,
            exploration_type='epsilon',
            qf_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            vf_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            policy_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            algorithm="TD3",

            dump_video_kwargs=dict(
                rows=1,
                columns=3,
            ),

            # do_state_exp=True,
            # algorithm='td3',
            # algo_kwargs=dict(
            #     num_epochs=100,
            #     max_path_length=50,
            #     batch_size=128,
            #     num_eval_steps_per_epoch=1000,
            #     num_expl_steps_per_train_loop=1000,
            #     num_trains_per_train_loop=1000,
            #     min_num_steps_before_training=10000,
            # ),
            # trainer_kwargs=dict(
            #     discount=0.99,
            # ),
            # replay_buffer_kwargs=dict(
            #     max_size=100000,
            #     fraction_goals_rollout_goals=0.2,
            #     fraction_goals_env_goals=0.0,
            # ),
            # qf_kwargs=dict(
            #     hidden_sizes=[400, 300],
            # ),
            # policy_kwargs=dict(
            #     hidden_sizes=[400, 300],
            # ),
        ),
    )
    # setup_logger('her-td3-sawyer-experiment', variant=variant)
    # experiment(variant)
    search_space = {
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    n_seeds = 1
    mode = 'local'
    exp_prefix = 'dev-{}'.format(
        __file__.replace('/', '-').replace('_', '-').split('.')[0]
    )

    # n_seeds = 5
    # mode = 'sss'
    # exp_prefix = 'railrl-her-sac-multiworld-sawyer-reach-min-num-steps'

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                #experiment,
                rl_experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                use_gpu=True,
                time_in_mins=1000,
          )
