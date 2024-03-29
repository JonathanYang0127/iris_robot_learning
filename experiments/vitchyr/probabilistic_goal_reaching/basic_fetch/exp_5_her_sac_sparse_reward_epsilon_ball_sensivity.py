import rlkit.util.hyperparameter as hyp
from rlkit.launchers.experiments.vitchyr.probabilistic_goal_reaching.launcher import \
    probabilistic_goal_reaching_experiment
from rlkit.launchers.launcher_util import run_experiment

if __name__ == "__main__":
    variant = dict(
        env_id='FetchPush-v1',
        observation_key='observation',
        desired_goal_key='desired_goal',
        # env_id='Point2DLargeEnv-v1',
        qf_kwargs=dict(
            hidden_sizes=[64, 64],
        ),
        policy_kwargs=dict(
            hidden_sizes=[64, 64],
        ),
        pgr_trainer_kwargs=dict(
            reward_scale=1,
            soft_target_tau=1e-3,
            target_update_period=1,
            use_automatic_entropy_tuning=True,
        ),
        discount_factor=0.99,
        reward_type='sparse',
        max_path_length=50,
        algo_kwargs=dict(
            batch_size=256,
            num_epochs=500,
            num_eval_steps_per_epoch=10000,
            num_expl_steps_per_train_loop=10000,
            num_trains_per_train_loop=10000,
            min_num_steps_before_training=10000,
            # num_epochs=5,
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
            rows=3,
            columns=2,
            subpad_length=0,
            subpad_color=127,
            pad_length=1,
            pad_color=0,
            num_columns_per_rollout=5,
            horizon=50,
        ),
        exploration_policy_kwargs=dict(
            exploration_version='occasionally_repeat',
            repeat_prob=0.5,
        ),
        video_renderer_kwargs=dict(
            width=256,
            height=256,
            output_image_format='CHW',
        ),
        learn_discount_model=False,
        plot_reward=True,
        plot_bootstrap_value=True,
        dynamics_adam_config=dict(
            lr=1e-2,
        ),
    )

    search_space = {
        'pgr_trainer_kwargs.reward_type': [
            'normal',
            # 'discounted',
            # 'discounted_plus_time_kl',
        ],
        'success_threshold': [
            0.0,
            0.01,
            0.05,
            0.1,
            0.2,
        ],
        'reward_type': [
            'sparse',
        ],
        'replay_buffer_kwargs': [
            dict(
                fraction_future_context=0.8,
                fraction_distribution_context=0.0,
                max_size=int(1e6),
            ),
        ],
        'exploration_policy_kwargs.repeat_prob': [
            0.5,
        ],
        'exploration_policy_kwargs.prob_random_action': [
            0.3,
        ],
        'exploration_policy_kwargs.exploration_version': [
            # 'occasionally_repeat',
            # 'epsilon_greedy',
            'epsilon_greedy_and_occasionally_repeat',
        ],
        'env_id': [
            # 'FetchPickAndPlace-v1',
            # 'FetchSlide-v1',
            'FetchPush-v1',
        ],
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
    exp_name = 'pgr--basic-fetch--exp-5--push-epsilon-ball-sensitivity-use-sparse-rewards'

    if mode == 'local':
        variant['algo_kwargs'] =dict(
            batch_size=32,
            num_epochs=1,
            num_eval_steps_per_epoch=100,
            num_expl_steps_per_train_loop=100,
            num_trains_per_train_loop=100,
            min_num_steps_before_training=100,
        )
        variant['save_video'] = True

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        # expl_params = variant['exploration_policy_kwargs']
        # if (
        #     expl_params['exploration_version'] == 'epsilon_greedy'
        #     and expl_params['repeat_prob'] != 0.
        # ):
        #     continue
        # if (
        #     expl_params['exploration_version'] == 'occasionally_repeat'
        #     and expl_params['prob_random_action'] != 0.
        # ):
        #     continue
        # if (
        #         expl_params['exploration_version'] == 'epsilon_greedy_and_occasionally_repeat'
        #         and not (
        #             expl_params['prob_random_action'] == 0.3
        #             and expl_params['repeat_prob'] == 0.5
        #         )
        # ):
        #     continue
        # print(expl_params)
        for seed in range(n_seeds):
            variant['exp_id'] = exp_id
            # variant['seed'] = seed
            run_experiment(
                probabilistic_goal_reaching_experiment,
                exp_name=exp_name,
                mode=mode,
                variant=variant,
                use_gpu=False,
                num_exps_per_instance=3,
                # slurm_config_name='cpu_co',
                gcp_kwargs=dict(
                    zone='us-east1-c',
                    gpu_kwargs=dict(
                        gpu_model='nvidia-tesla-k80',
                        num_gpu=1,
                    )
                ),
                time_in_mins=int(2.5*24*60),
            )
