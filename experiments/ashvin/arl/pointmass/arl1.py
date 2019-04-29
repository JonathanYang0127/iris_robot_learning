import railrl.misc.hyperparameter as hyp
from experiments.murtaza.multiworld.skew_fit.reacher.generate_uniform_dataset import generate_uniform_dataset_reacher
from multiworld.envs.mujoco.cameras import sawyer_init_camera_zoomed_in
from railrl.launchers.launcher_util import run_experiment
# from railrl.torch.grill.launcher import grill_her_twin_sac_online_vae_full_experiment
from railrl.torch.grill.launcher import *
import railrl.torch.vae.vae_schedules as vae_schedules
from railrl.torch.vae.conv_vae import imsize48_default_architecture
from railrl.launchers.arglauncher import run_variants

from multiworld.envs.pygame.point2d import Point2DWallEnv

from railrl.torch.arl.models.hinge_distance_model_trainer import HingeDistanceModelTrainer

from railrl.torch.networks import CNN

# def experiment(variant):
#     full_experiment_variant_preprocess(variant)
#     train_vae_and_update_variant(variant)

if __name__ == "__main__":
    variant = dict(
        double_algo=False,
        online_vae_exploration=False,
        imsize=48,
        init_camera=sawyer_init_camera_zoomed_in,
        # env_id='SawyerPushNIPSEasy-v0',

        env_class=Point2DWallEnv,
        env_kwargs=dict(
            render_onscreen=False,
            ball_radius=1,
            images_are_rgb=True,
            show_goal=False,
        ),

        grill_variant=dict(
            save_video=True,
            custom_goal_sampler='replay_buffer',
            model_class=CNN,
            model_kwargs=dict(
                kernel_sizes=[5, 3, 3],
                n_channels=[16, 32, 64],
                strides=[3, 2, 2],
                hidden_sizes=[],
                batch_norm_conv=False,
                batch_norm_fc=False,
                paddings=[0, 0, 0],
                input_height=48,
                input_width=48,
                input_channels=3,
                output_size=4,
            ),
            model_trainer_class=HingeDistanceModelTrainer,
            model_trainer_kwargs=dict(
                beta=20,
                lr=1e-3,
            ),
            save_video_period=100,
            qf_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            policy_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            vf_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            max_path_length=50,
            algo_kwargs=dict(
                batch_size=1024,
                num_epochs=1000,
                num_eval_steps_per_epoch=500,
                num_expl_steps_per_train_loop=500,
                num_trains_per_train_loop=1000,
                min_num_steps_before_training=10000,
                model_training_schedule=vae_schedules.custom_schedule_2,
                oracle_data=False,
                vae_save_period=1,
            ),
            twin_sac_trainer_kwargs=dict(
                discount=0.99,
                reward_scale=1,
                soft_target_tau=1e-3,
                target_update_period=1,  # 1
                use_automatic_entropy_tuning=True,
            ),
            replay_buffer_kwargs=dict(
                max_size=100000,
                ob_keys_to_save=["image_observation"],
            ),
            exploration_goal_sampling_mode='vae_prior',
            evaluation_goal_sampling_mode='reset_of_env',
            normalize=False,
            render=False,
            exploration_noise=0.0,
            exploration_type='ou',
            training_mode='train',
            testing_mode='test',
            reward_params=dict(
                type='latent_distance',
            ),
            observation_key='latent_observation',
            desired_goal_key='latent_desired_goal',
            vae_wrapped_env_kwargs=dict(
                sample_from_true_prior=True,
            ),
            algorithm='ONLINE-VAE-SAC-BERNOULLI',
            # generate_uniform_dataset_kwargs=dict(
                # init_camera=sawyer_init_camera_zoomed_in,
                # env_id='SawyerPushNIPS-v0',
                # num_imgs=1000,
                # use_cached_dataset=False,
                # show=False,
                # save_file_prefix='pusher',
            # ),
            # generate_uniform_dataset_fn=generate_uniform_dataset_reacher,
        ),
        train_vae_variant=dict(
            representation_size=4,
            beta=20,
            num_epochs=0,
            dump_skew_debug_plots=False,
            # decoder_activation='gaussian',
            decoder_activation='sigmoid',
            generate_vae_dataset_kwargs=dict(
                N=100,
                test_p=.9,
                use_cached=False,
                show=False,
                oracle_dataset=True,
                oracle_dataset_using_set_to_goal=True,
                n_random_steps=100,
                non_presampled_goal_img_is_garbage=True,
            ),
            vae_kwargs=dict(
                input_channels=3,
                architecture=imsize48_default_architecture,
                decoder_distribution='gaussian_identity_variance',
            ),
            # TODO: why the redundancy?
            algo_kwargs=dict(
                start_skew_epoch=5000,
                is_auto_encoder=False,
                batch_size=64,
                lr=1e-3,
                skew_config=dict(
                    method='vae_prob',
                    power=0,
                ),
                skew_dataset=False,
                priority_function_kwargs=dict(
                    decoder_distribution='gaussian_identity_variance',
                    sampling_method='importance_sampling',
                    # sampling_method='true_prior_sampling',
                    num_latents_to_sample=10,
                ),
                use_parallel_dataloading=False,
            ),

            save_period=25,
        ),
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

    run_variants(arl_full_experiment, variants, run_id=1)
