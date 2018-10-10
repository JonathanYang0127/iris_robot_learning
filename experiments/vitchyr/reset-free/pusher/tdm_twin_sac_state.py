import railrl.misc.hyperparameter as hyp
from railrl.launchers.experiments.vitchyr.multiworld import (
    tdm_twin_sac_experiment,
)
from railrl.launchers.launcher_util import run_experiment

if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algo_kwargs=dict(
            base_kwargs=dict(
                num_epochs=1000,
                num_steps_per_epoch=1000,
                num_steps_per_eval=5000,
                max_path_length=500,
                num_updates_per_env_step=1,
                batch_size=128,
                discount=1,
                min_num_steps_before_training=1000,
                reward_scale=100,
                render=False,
            ),
            tdm_kwargs=dict(),
            twin_sac_kwargs=dict(),
        ),
        env_id='SawyerPushAndReachXYEnv-ResetFree-v0',
        replay_buffer_kwargs=dict(
            max_size=int(1E6),
            fraction_goals_are_rollout_goals=0.2,
            fraction_resampled_goals_are_env_goals=0.5,
        ),
        qf_kwargs=dict(
            hidden_sizes=[400, 300],
            structure='none',
        ),
        vf_kwargs=dict(
            hidden_sizes=[400, 300],
            structure='none',
        ),
        policy_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        algorithm='TDM-TwinSAC',
        version='normal',
        es_kwargs=dict(
            max_sigma=.8,
        ),
        exploration_type='ou',
        save_video_period=100,
        do_state_exp=True,
        imsize=48,
        save_video=False,
        observation_key='state_observation',
        desired_goal_key='state_desired_goal',
        vectorized=False,
    )
    search_space = {
        'env_id': [
            'SawyerPushXYEnv-CompleteResetFree-v1',
            # 'SawyerPushAndReachXYEnv-CompleteResetFree-v0',
            'SawyerPushXYEnv-WithResets-v0',
            # 'SawyerPushAndReachXYEnv-WithResets-v0',
        ],
        # 'algo_kwargs.tdm_kwargs.max_tau': [
        #     100, 50, 30,
        # ],
        'algo_kwargs.tdm_kwargs.dense_rewards': [
            True,
        ],
        'algo_kwargs.tdm_kwargs.finite_horizon': [
            False,
        ],
        'algo_kwargs.base_kwargs.discount': [
            0.99,
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    n_seeds = 1
    mode = 'local'
    exp_prefix = 'dev'

    n_seeds = 5
    mode = 'sss'
    exp_prefix = 'push-tdm-tsac-her-settings-test-no-structure'

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for i in range(n_seeds):
            run_experiment(
                tdm_twin_sac_experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                time_in_mins=23*60,
                snapshot_mode='gap_and_last',
                snapshot_gap=100,
            )
