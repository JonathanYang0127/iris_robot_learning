import rlkit.util.hyperparameter as hyp
from rlkit.demos.source.encoder_dict_to_mdp_path_loader import EncoderDictToMDPPathLoader
from rlkit.launchers.experiments.ashvin.awac_rig import awac_rig_experiment, process_args
from rlkit.launchers.launcher_util import run_experiment
from rlkit.launchers.arglauncher import run_variants
from rlkit.torch.sac.policies import GaussianPolicy, GaussianMixturePolicy, GaussianCNNPolicy, GaussianTwoChannelCNNPolicy
from rlkit.torch.networks.cnn import ConcatCNN, CNN
from roboverse.envs.sawyer_rig_multiobj_v0 import SawyerRigMultiobjV0
from roboverse.envs.sawyer_rig_multiobj_tray_v0 import SawyerRigMultiobjTrayV0
from roboverse.envs.sawyer_rig_affordances_v0 import SawyerRigAffordancesV0
from roboverse.envs.sawyer_rig_affordances_v1 import SawyerRigAffordancesV1
from rlkit.torch.networks import Clamp
from rlkit.torch.vae.vq_vae import VQ_VAE
from rlkit.torch.vae.vq_vae_trainer import VQ_VAETrainer
from rlkit.torch.grill.common import train_vqvae

DATASETS = ["val", "reset-free", "tray-reset-free", "tray-test-reset-free", "rotated-top-drawer-reset-free"]

dataset = "rotated-top-drawer-reset-free"
assert dataset in DATASETS

# VAL Data
if dataset == 'val':
    VAL_DATA_PATH = "data/combined/"
    demo_paths=[dict(path=VAL_DATA_PATH + 'drawer_demos_0.pkl', obs_dict=True, is_demo=True),
                dict(path=VAL_DATA_PATH + 'drawer_demos_1.pkl', obs_dict=True, is_demo=True),
                dict(path=VAL_DATA_PATH + 'pnp_demos_0.pkl', obs_dict=True, is_demo=True),
                dict(path=VAL_DATA_PATH + 'tray_demos_0.pkl', obs_dict=True, is_demo=True),

                dict(path=VAL_DATA_PATH + 'drawer_demos_2.pkl', obs_dict=True, is_demo=True),
                dict(path=VAL_DATA_PATH + 'drawer_demos_3.pkl', obs_dict=True, is_demo=True),
                dict(path=VAL_DATA_PATH + 'pnp_demos_1.pkl', obs_dict=True, is_demo=True),
                dict(path=VAL_DATA_PATH + 'tray_demos_1.pkl', obs_dict=True, is_demo=True),

                dict(path=VAL_DATA_PATH + 'drawer_demos_4.pkl', obs_dict=True, is_demo=True),
                dict(path=VAL_DATA_PATH + 'drawer_demos_5.pkl', obs_dict=True, is_demo=True),
                dict(path=VAL_DATA_PATH + 'pnp_demos_2.pkl', obs_dict=True, is_demo=True),
                dict(path=VAL_DATA_PATH + 'tray_demos_2.pkl', obs_dict=True, is_demo=True),

                dict(path=VAL_DATA_PATH + 'drawer_demos_6.pkl', obs_dict=True, is_demo=True),
                dict(path=VAL_DATA_PATH + 'drawer_demos_7.pkl', obs_dict=True, is_demo=True),
                dict(path=VAL_DATA_PATH + 'pnp_demos_3.pkl', obs_dict=True, is_demo=True),
                dict(path=VAL_DATA_PATH + 'tray_demos_3.pkl', obs_dict=True, is_demo=True),
                ]
# Reset-Free Data
elif dataset == 'reset-free':
    VAL_DATA_PATH = "data/combined_reset_free_v5/"
    demo_paths=[dict(path=VAL_DATA_PATH + 'combined_reset_free_v5_demos_{}.pkl'.format(str(i)), obs_dict=True, is_demo=True, use_latents=True) for i in range(16)]
# Tray Reset-Free Data (with distractors)
elif dataset == 'tray-reset-free':
    VAL_DATA_PATH = "data/combined_reset_free_v5_tray_only/"
    demo_paths=[dict(path=VAL_DATA_PATH + 'reset_free_v5_tray_only_demos_{}.pkl'.format(str(i)), obs_dict=True, is_demo=True, use_latents=True) for i in range(16)]
# Tray Reset-Free Data (without distractors)
elif dataset == 'tray-test-reset-free':
    VAL_DATA_PATH = "data/combined_reset_free_v5_tray_test_env_only/"
    demo_paths=[dict(path=VAL_DATA_PATH + 'reset_free_v5_tray_test_env_only_demos_{}.pkl'.format(str(i)), obs_dict=True, is_demo=True, use_latents=True) for i in range(16)]
# Rotated Drawer Reset-Free Data
elif dataset == "rotated-top-drawer-reset-free":
    VAL_DATA_PATH = "data/reset_free_v5_rotated_top_drawer/"
    demo_paths=[dict(path=VAL_DATA_PATH + 'reset_free_v5_rotated_top_drawer_demos_{}.pkl'.format(str(i)), obs_dict=True, is_demo=True, use_latents=True) for i in range(16)]
else:
    assert False

vqvae = VAL_DATA_PATH + "best_vqvae.pt"
image_train_data = VAL_DATA_PATH + 'combined_images.npy'
image_test_data = VAL_DATA_PATH + 'combined_test_images.npy'

if __name__ == "__main__":
    variant = dict(
        imsize=48,
        env_kwargs=dict(
            test_env=True,
        ),
        policy_class=GaussianPolicy,
        policy_kwargs=dict(
            hidden_sizes=[256, 256, 256, 256],
            max_log_std=0,
            min_log_std=-6,
            std_architecture="values",
        ),
        qf_kwargs=dict(
            hidden_sizes=[256, 256],
        ),

        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            beta=1,
            use_automatic_entropy_tuning=False,
            alpha=0,

            policy_weight_decay=1e-4,
            q_weight_decay=0,

            rl_weight=1.0,
            use_awr_update=True,
            use_reparam_update=False,
            compute_bc=False,
            reparam_weight=0.0,
            awr_weight=1.0,
            bc_weight=0.0,

            reward_transform_kwargs=None,
            terminal_transform_kwargs=None,
        ),

        max_path_length=65, #50
        algo_kwargs=dict(
            batch_size=1024,
            start_epoch=-25, # offline epochs
            num_epochs=1001, # online epochs
            num_eval_steps_per_epoch=1000,
            num_expl_steps_per_train_loop=1000, 
            num_trains_per_train_loop=1000,
            min_num_steps_before_training=4000,
        ),
        replay_buffer_kwargs=dict(
            fraction_future_context=0.6,
            fraction_distribution_context=0.1,
            max_size=int(1E6),
        ),
        online_offline_split_replay_buffer_kwargs=dict(
            online_replay_buffer_kwargs=dict(
                fraction_future_context=0.6,
                fraction_distribution_context=0.1,
                max_size=int(4E5),
            ),
            offline_replay_buffer_kwargs=dict(
                fraction_future_context=0.6,
                fraction_distribution_context=0.1,
                max_size=int(6E5),
            ),
            sample_online_fraction=0.2
        ),
        demo_replay_buffer_kwargs=dict(
            fraction_future_context=0.6,
            fraction_distribution_context=0.1,
        ),
        reward_kwargs=dict(
            reward_type='sparse',
            epsilon=1.0,
        ),

        observation_key='latent_observation',
        observation_keys=['latent_observation'],
        desired_goal_key='latent_desired_goal',
        save_video=True,
        save_video_kwargs=dict(
            save_video_period=25,
            pad_color=0,
        ),

        reset_keys_map=dict(
            image_observation="initial_latent_state"
        ),
        pretrained_vae_path=vqvae,

        path_loader_class=EncoderDictToMDPPathLoader,
        path_loader_kwargs=dict(
            delete_after_loading=True,
            recompute_reward=True,
            demo_paths=demo_paths,
        ),

        renderer_kwargs=dict(
            create_image_format='HWC',
            output_image_format='CWH',
            flatten_image=True,
            width=48,
            height=48,
        ),

        add_env_demos=False,
        add_env_offpolicy_data=False,

        load_demos=True,

        evaluation_goal_sampling_mode="presampled_images",
        exploration_goal_sampling_mode="conditional_vae_prior",#"presampled_images",#"presample_latents",
        training_goal_sampling_mode="presample_latents",#"presampled_images",

        train_vae_kwargs=dict(
            imsize=48,
            beta=1,
            beta_schedule_kwargs=dict(
                x_values=(0, 250),
                y_values=(0, 100),
            ),
            num_epochs=1501, #1501
            embedding_dim=5,
            dump_skew_debug_plots=False,
            decoder_activation='sigmoid',
            use_linear_dynamics=False,
            generate_vae_dataset_kwargs=dict(
                N=1000,
                n_random_steps=2,
                test_p=.9,
                dataset_path={'train': image_train_data,
                              'test': image_test_data,
                              },
                augment_data=False,
                use_cached=False,
                show=False,
                oracle_dataset=False,
                oracle_dataset_using_set_to_goal=False,
                non_presampled_goal_img_is_garbage=False,
                delete_after_loading=True,
                random_rollout_data=True,
                random_rollout_data_set_to_goal=True,
                conditional_vae_dataset=True,
                save_trajectories=False,
                enviorment_dataset=False,
                tag="ccrig_tuning_orig_network",
            ),
            vae_trainer_class=VQ_VAETrainer,
            vae_class=VQ_VAE,
            vae_kwargs=dict(
                input_channels=3,
                imsize=48,
            ),
            algo_kwargs=dict(
                key_to_reconstruct='x_t',
                start_skew_epoch=5000,
                is_auto_encoder=False,
                batch_size=128,
                lr=1e-3,
                skew_config=dict(
                    method='vae_prob',
                    power=0,
                ),
                weight_decay=0.0,
                skew_dataset=False,
                priority_function_kwargs=dict(
                    decoder_distribution='gaussian_identity_variance',
                    sampling_method='importance_sampling',
                    num_latents_to_sample=10,
                ),
                use_parallel_dataloading=False,
            ),

            save_period=50,
        ),
        train_model_func=train_vqvae,
        presampled_goal_kwargs=dict(
            eval_goals='', #HERE
            expl_goals='',
            training_goals='',
        ),
        launcher_config=dict(
            unpack_variant=True,
            region='us-west-1', #HERE
        ),
    )

    search_space = {
        "seed": range(3),
        'env_type': ['top_drawer', 'bottom_drawer', 'tray', 'pnp'],
        'reward_kwargs.epsilon': [4.0], #3.5, 4.0, 4.5, 5.0, 5.5, 6.0
        'env_kwargs.reset_interval' : [1],#[1, 2, 4, 5, 10, 15, 20, 25],
        #'tray_obj' : ['mug', 'long_sofa', 'camera', 'grill_trash_can', 'beer_bottle'], # Set goal

        "image": [False], # Latent-space or image-space
        "online_offline_split": [False], # Single replay buffer vs Two replay buffers (one for online, one for offline)
        "ground_truth_expl_goals": [False], # PixelCNN expl goals vs ground truth expl goals

        'trainer_kwargs.beta': [0.3],
        # 'num_pybullet_objects':[None],
        'policy_kwargs.min_log_std': [-6],
        'trainer_kwargs.awr_weight': [1.0],
        'trainer_kwargs.awr_use_mle_for_vf': [True, ],
        'trainer_kwargs.awr_sample_actions': [False, ],
        'trainer_kwargs.clip_score': [2, ],
        'trainer_kwargs.awr_min_q': [True, ],
        'trainer_kwargs.reward_transform_kwargs': [None, ],
        'trainer_kwargs.terminal_transform_kwargs': [dict(m=0, b=0),],
        'qf_kwargs.output_activation': [Clamp(max=0)],
        'replay_buffer_kwargs.max_size': [int(1E6)],
    }

    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    variants = []
    for variant in sweeper.iterate_hyperparameters():
        env_type = variant['env_type']
        if dataset != 'val' and env_type == 'pnp':
            env_type = 'obj'
        if 'tray_obj' in variant and env_type == 'tray':
            eval_goals = EVAL_DATA_PATH + 'tray_{0}_goals.pkl'.format(variant['tray_obj'])
            variant['env_kwargs']['test_object_subset'] = [variant['tray_obj']]
        else:
            eval_goals = EVAL_DATA_PATH + '{0}_goals.pkl'.format(env_type)
        variant['presampled_goal_kwargs']['eval_goals'] = eval_goals

        if variant['ground_truth_expl_goals']:
            variant['exploration_goal_sampling_mode']="presampled_images" #"presample_latents"
            variant['training_goal_sampling_mode']="presampled_images"
            variant['presampled_goal_kwargs']['expl_goals'] = eval_goals
            variant['presampled_goal_kwargs']['training_goals'] = eval_goals

        if dataset == 'val':
            if env_type in ['top_drawer', 'bottom_drawer']:
                variant['env_class'] = SawyerRigAffordancesV0
                variant['env_kwargs']['env_type'] = env_type
            if env_type == 'tray':
                variant['env_class'] = SawyerRigMultiobjTrayV0
            if env_type == 'pnp':
                variant['env_class'] = SawyerRigMultiobjV0
        elif dataset in ["reset-free", "tray-reset-free", "tray-test-reset-free"]:
            variant['env_class'] = SawyerRigAffordancesV0
            variant['env_kwargs']['env_type'] = env_type
        elif dataset == "rotated-top-drawer-reset-free":
            variant['env_class'] = SawyerRigAffordancesV1
        else:
            assert False

        #Image
        if variant['image']:
            variant['policy_class'] = GaussianCNNPolicy
            variant['qf_class'] = ConcatCNN

            variant['policy_kwargs'] = dict(
                # CNN params
                input_width=48,
                input_height=48,
                input_channels=6,
                kernel_sizes=[3, 3, 3],
                n_channels=[16, 16, 16],
                strides=[1, 1, 1],
                hidden_sizes=[1024, 512, 256],
                paddings=[1, 1, 1],
                pool_type='max2d',
                pool_sizes=[2, 2, 1],  # the one at the end means no pool
                pool_strides=[2, 2, 1],
                pool_paddings=[0, 0, 0],
                # Gaussian params
                max_log_std=0,
                min_log_std=-6,
                std_architecture="values",
            )
            variant['qf_kwargs'] = dict(
                input_width=48,
                input_height=48,
                input_channels=6,
                kernel_sizes=[3, 3, 3],
                n_channels=[16, 16, 16],
                strides=[1, 1, 1],
                hidden_sizes=[1024, 512, 256],
                paddings=[1, 1, 1],
                pool_type='max2d',
                pool_sizes=[2, 2, 1],  # the one at the end means no pool
                pool_strides=[2, 2, 1],
                pool_paddings=[0, 0, 0],
            )

            ## Keys used by reward function for reward calculation
            variant['observation_key_reward_fn'] = 'latent_observation'
            variant['desired_goal_key_reward_fn'] = 'latent_desired_goal'

            ## Keys used by policy/q-networks
            variant['observation_key'] = 'image_observation'
            variant['desired_goal_key'] = 'image_desired_goal'

            for demo_path in variant['path_loader_kwargs']['demo_paths']:
                demo_path['use_latents'] = False

        variants.append(variant)

    run_variants(awac_rig_experiment, variants, run_id=0, process_args_fn=process_args)