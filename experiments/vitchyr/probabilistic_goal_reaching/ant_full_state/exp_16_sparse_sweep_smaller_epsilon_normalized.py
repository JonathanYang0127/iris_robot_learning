import rlkit.util.hyperparameter as hyp
from rlkit.launchers.experiments.vitchyr.probabilistic_goal_reaching.launcher import \
    probabilistic_goal_reaching_experiment
from rlkit.launchers.launcher_util import run_experiment

if __name__ == "__main__":
    variant = dict(
        # env_id='FetchPush-v1',
        observation_key='observation',
        desired_goal_key='desired_goal',
        # env_id='Point2DLargeEnv-v1',
        qf_kwargs=dict(
            hidden_sizes=[400, 300],
            # hidden_sizes=[64, 64],
        ),
        policy_kwargs=dict(
            hidden_sizes=[400, 300],
            # hidden_sizes=[64, 64],
        ),
        pgr_trainer_kwargs=dict(
            reward_scale=1,
            soft_target_tau=1e-3,
            target_update_period=1,
            use_automatic_entropy_tuning=True,
        ),
        discount_factor=0.99,
        reward_type='log_prob',
        max_path_length=50,
        algo_kwargs=dict(
            batch_size=256,
            num_epochs=300,
            num_eval_steps_per_epoch=10000,
            num_expl_steps_per_train_loop=10000,
            num_trains_per_train_loop=10000,
            min_num_steps_before_training=10000,
            # num_epochs=1,
            # num_eval_steps_per_epoch=100,
            # num_expl_steps_per_train_loop=100,
            # num_trains_per_train_loop=100,
            # min_num_steps_before_training=100,
        ),
        dynamics_model_version='fixed_standard_gaussian',
        dynamics_model_config=dict(
            hidden_sizes=[64, 64],
        ),
        replay_buffer_kwargs=dict(
            fraction_future_context=0.5,
            fraction_distribution_context=0.5,
            max_size=int(1e6),
        ),
        save_video=True,
        save_video_kwargs=dict(
            save_video_period=20,
            rows=4,
            columns=1,
            subpad_length=0,
            subpad_color=127,
            pad_length=1,
            pad_color=0,
            num_columns_per_rollout=5,
            horizon=100,
        ),
        exploration_policy_kwargs=dict(
            exploration_version='epsilon_greedy_and_occasionally_repeat',
            repeat_prob=0.,
            prob_random_action=0.,
        ),
        video_renderer_kwargs=dict(
            width=256,
            height=256,
            output_image_format='CHW',
        ),
        learn_discount_model=False,
        visualize_all_plots=False,
        plot_reward=True,
        plot_bootstrap_value=True,
        dynamics_adam_config=dict(
            lr=1e-2,
        ),
    )

    search_space = {
        'env_id': [
            # 'Ant-v1',
            # 'LowGearAntXY-v0',
            'AntFullPositionGoal-v0',
            # 'AntXY-NoContactSensors-v0',
            # 'AntXY-LowGear-v0',
            # 'AntXY-LowGear-NoContactSensors-v0',
        ],
        'dynamics_model_version': [
            # 'learned_model_ensemble',
            # 'learned_model',
            # 'learned_model_laplace',
            'fixed_standard_laplace',
        ],
        'pgr_trainer_kwargs.reward_type': [
            'normal',
        ],
        'pgr_trainer_kwargs.discount_type': [
            'prior',
        ],
        'reward_type': [
            # 'log_prob',
            'sparse',
            # 'negative_distance',
            # 'prob',
        ],
        'success_threshold': [
            0.0,
            0.05,
            0.25,
            1.0,
            2.0,
        ],
        'replay_buffer_kwargs': [
            # dict(
            #     fraction_future_context=0.5,
            #     fraction_distribution_context=0.0,
            #     fraction_next_context=0.25,
            #     max_size=int(1e6),
            # ),
            dict(
                fraction_future_context=0.8,
                fraction_distribution_context=0.0,
                fraction_next_context=0.,
                max_size=int(1e6),
            ),
            # dict(
            #     fraction_future_context=0.5,
            #     fraction_distribution_context=0.5,
            #     max_size=int(1e6),
            # ),
        ],
        'pgr_trainer_kwargs.reward_scale': [
            # 'auto_normalize_by_mean_magnitude',
            # 'auto_normalize_by_max_magnitude',
            # 0.01,
            1,
        ],
        'max_path_length': [
            100,
        ],
        # 'normalize_distances_for_full_state_ant': [True, False],
        'normalize_distances_for_full_state_ant': [True],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    n_seeds = 1
    mode = 'local'
    exp_name = 'dev-{}'.format(
        __file__.replace('/', '-').replace('_', '-').split('.')[0]
    )

    n_seeds = 4
    mode = 'htp'
    exp_name = 'pgr--ant-full-state--exp-16--sparse-sweep-smaller-epsilon-normalized'

    if mode == 'local':
        variant['algo_kwargs'] =dict(
            batch_size=32,
            num_epochs=4,
            num_eval_steps_per_epoch=100,
            num_expl_steps_per_train_loop=100,
            num_trains_per_train_loop=100,
            min_num_steps_before_training=100,
        )
        variant['save_video'] = True

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for seed in range(n_seeds):
            variant['exp_id'] = exp_id
            # variant['seed'] = seed
            run_experiment(
                probabilistic_goal_reaching_experiment,
                exp_name=exp_name,
                mode=mode,
                variant=variant,
                use_gpu=False,
                num_exps_per_instance=2,
                # slurm_config_name='cpu_co',
                gcp_kwargs=dict(
                    zone='us-east1-c',
                    gpu_kwargs=dict(
                        gpu_model='nvidia-tesla-k80',
                        num_gpu=1,
                    )
                ),
                time_in_mins=int(2.9*24*60),
            )
