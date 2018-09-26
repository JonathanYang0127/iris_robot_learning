import railrl.misc.hyperparameter as hyp
from multiworld.envs.mujoco.cameras import sawyer_door_env_camera_v3
from multiworld.envs.mujoco.sawyer_xyz.sawyer_door_hook import SawyerDoorHookEnv
from railrl.launchers.experiments.murtaza.multiworld import her_td3_experiment
from railrl.launchers.launcher_util import run_experiment
from railrl.torch.vae.generate_goal_dataset import generate_goal_dataset_using_policy

if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algo_kwargs=dict(
            base_kwargs=dict(
                num_epochs=500,
                num_steps_per_epoch=1000,
                num_steps_per_eval=500,
                max_path_length=100,
                num_updates_per_env_step=1,
                batch_size=128,
                discount=0.99,
                min_num_steps_before_training=4000,
                reward_scale=100,
                render=False,
                collection_mode='online-parallel',
                parallel_env_params=dict(
                    num_workers=1,
                )

            ),
            her_kwargs=dict(
                observation_key='state_observation',
                desired_goal_key='state_desired_goal',
            ),
            td3_kwargs=dict(),
        ),
        replay_buffer_kwargs=dict(
            max_size=int(1E6),
            fraction_goals_are_rollout_goals=0.2,
            fraction_resampled_goals_are_env_goals=0.5,
        ),
        qf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        policy_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        algorithm='HER-TD3',
        version='normal',
        es_kwargs=dict(
            max_sigma=.8,
        ),
        exploration_type='ou',
        observation_key='state_observation',
        desired_goal_key='state_desired_goal',
        init_camera=sawyer_door_env_camera_v3,
        env_id='SawyerDoorHookResetFreeEnv-v4',
        # env_class=SawyerDoorHookEnv,
        # env_kwargs=dict(
        #     goal_low=(-0.25, 0.45, 0.1, 0),
        #     goal_high=(0.25, 0.65, .25, 1.0472),
        #     hand_low=(-0.25, 0.45, 0.15),
        #     hand_high=(.25, 0.65, .25),
        #     max_angle=1.0472,
        #     xml_path='sawyer_xyz/sawyer_door_pull_hook.xml',
        #     reset_free=True,
        # ),
        imsize=48,
        do_state_exp=True,
        save_video_period=50,
        save_video=False,
        presample_goals=True,
        presampled_goals_path='goals/sawyer_hook_door_goals.npy',
    )
    search_space = {
        'algo_kwargs.base_kwargs.max_path_length': [100, 200, 500],
        'env_kwargs.reward_type': [
            'angle_diff_and_hand_distance',
        ],
        'env_kwargs.reset_free':[True],
        'env_kwargs.target_pos_scale':[1]
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    # n_seeds = 1
    # mode = 'local'
    # exp_prefix = 'test'

    n_seeds = 1
    mode = 'ec2'
    exp_prefix = 'sawyer_door_new_door_60_reset_free_hard_space-v2_path_length_sweep'

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for i in range(n_seeds):
            run_experiment(
                her_td3_experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                snapshot_mode='gap_and_last',
                snapshot_gap=50,
                variant=variant,
                use_gpu=True,
                num_exps_per_instance=3,
            )
