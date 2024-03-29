from rlkit.launchers.arglauncher import run_variants
import rlkit.util.hyperparameter as hyp
from multiworld.envs.pygame import PickAndPlaceEnv
from rlkit.launchers.experiments.goal_distribution.irl_launcher import \
    representation_learning_with_goal_distribution_launcher

from rlkit.launchers.launcher_util import run_experiment
# from rlkit.torch.sets.launcher import test_offline_set_vae
# from rlkit.launchers.masking_launcher import default_masked_reward_fn
from rlkit.envs.contextual.mask_conditioned import default_masked_reward_fn
from rlkit.launchers.exp_launcher import rl_context_experiment

from roboverse.envs.goal_conditioned.sawyer_lift_gc import SawyerLiftEnvGC

if __name__ == '__main__':
    imsize = 48
    variant = dict(
        algo_kwargs=dict(
            num_epochs=2501,
            batch_size=2048,
            num_eval_steps_per_epoch=1000,
            num_expl_steps_per_train_loop=1000,
            num_trains_per_train_loop=1000, #4000,
            min_num_steps_before_training=1000,
            eval_epoch_freq=1,
        ),
        max_path_length=100,
        sac_trainer_kwargs=dict(
            soft_target_tau=1e-3,
            target_update_period=1,
            use_automatic_entropy_tuning=True,
            reward_scale=100,
            discount=0.99,
        ),
        contextual_replay_buffer_kwargs=dict(
            max_size=int(1E6),
            fraction_future_context=0.0,
            fraction_distribution_context=0.0,
            fraction_replay_buffer_context=0.0,
            # recompute_rewards=True,
        ),
        qf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        policy_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        observation_key='state_observation',
        desired_goal_key='state_desired_goal',
        achieved_goal_key='state_achieved_goal',
        expl_goal_sampling_mode='obj_in_bowl',
        eval_goal_sampling_mode='obj_in_bowl',
        save_env_in_snapshot=False,
        save_video=True,
        dump_video_kwargs=dict(
            rows=1,
            columns=8,
            pad_color=0,
            pad_length=0,
            subpad_length=1,
        ),
        save_video_period=150,
        renderer_kwargs=dict(
            width=imsize,
            height=imsize,
        ),
        mask_conditioned=False,
        mask_variant=dict(
            rollout_mask_order_for_expl='random',
            rollout_mask_order_for_eval='fixed',
            log_mask_diagnostics=True,
            mask_format='matrix',
            infer_masks=False,
            mask_inference_variant=dict(
                n=100,
                noise=0.01,
                max_cond_num=1e2,
                normalize_sigma_inv=True,
                sigma_inv_entry_threshold=0.10,
            ),
            relabel_goals=True,
            relabel_masks=True,
            sample_masks_for_relabeling=True,

            context_post_process_mode=None,
            context_post_process_frac=0.5,

            max_subtasks_to_focus_on=None,
            max_subtasks_per_rollout=None,
            prev_subtask_weight=0.25,
            reward_fn=default_masked_reward_fn,
            use_g_for_mean=True,

            train_mask_distr=dict(
                atomic=1.0,
                subset=0.0,
                cumul=0.0,
                full=0.0,
            ),
            expl_mask_distr=dict(
                atomic=0.5,
                atomic_seq=0.5,
                cumul_seq=0.0,
                full=0.0,
            ),
            eval_mask_distr=dict(
                atomic=0.0,
                atomic_seq=1.0,
                cumul_seq=0.0,
                full=0.0,
            ),
            idx_masks=[
                {2: 2, 3: 3},
                {4: 4, 5: 5},
            ],
            eval_rollouts_to_log=['atomic', 'atomic_seq'],
            eval_rollouts_for_videos=[],
        ),
        env_class=SawyerLiftEnvGC,
        env_kwargs={
            'action_scale': .06,
            'action_repeat': 10,
            'timestep': 1./120,
            'solver_iterations': 500,
            'max_force': 1000,

            'gui': False,
            'pos_init': [.75, -.3, 0],
            'pos_high': [.75, .4, .3],
            'pos_low': [.75, -.4, -.36],
            'reset_obj_in_hand_rate': 0.0,
            'bowl_bounds': [-0.40, 0.40],

            'use_rotated_gripper': True,
            'use_wide_gripper': True,
            'soft_clip': True,
            'obj_urdf': 'spam',
            'max_joint_velocity': None,

            'hand_reward': True,
            'gripper_reward': True,
            'bowl_reward': True,

            'goal_sampling_mode': 'obj_in_bowl',
            'random_init_bowl_pos': True,
            'bowl_type': 'heavy',

            'num_obj': 4,
        },
        logger_config=dict(
            snapshot_gap=25,
            snapshot_mode='gap_and_last',
        ),
        launcher_config=dict(
            unpack_variant=True,
        ),
        reward_trainer_kwargs=dict(
            mixup_alpha=0.5,
            data_split=0.1, # use 10% = 100 examples as data
            train_split=0.3, # use 30% of data = 30 examples for train
        ),
        # example_set_path="ashvin/lha/example_set_gen/07-22-pb-abs-example-set/07-22-pb-abs-example-set_2020_07_22_18_35_52_id000--s57269/example_dataset.npy",
        example_set_path="ashvin/lha/example_set_gen/07-22-pb-rel-example-set/07-22-pb-rel-example-set_2020_07_22_18_36_47_id000--s21183/example_dataset.npy",
        # example_set_path="ashvin/lha/example_set_gen/07-22-pg-example-set/07-22-pg-example-set_2020_07_22_18_35_29_id000--s6012/example_dataset.npy",
    )

    search_space = {
        'seedid': range(5),
    }

    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    variants = []
    for variant in sweeper.iterate_hyperparameters():
        variants.append(variant)

    run_variants(representation_learning_with_goal_distribution_launcher, variants, run_id=0)
