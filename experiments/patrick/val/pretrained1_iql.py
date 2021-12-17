import rlkit.util.hyperparameter as hyp
from rlkit.demos.source.encoder_dict_to_mdp_path_loader import EncoderDictToMDPPathLoader
from rlkit.launchers.experiments.ashvin.iql_rig import iql_rig_experiment, process_args
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

import itertools

DATASETS = [
    "val", "reset-free", "tray-reset-free", "tray-test-reset-free", "rotated-top-drawer-reset-free", 
    "reconstructed-rotated-top-drawer-reset-free", "antialias-rotated-top-drawer-reset-free",
    "antialias-right-top-drawer-reset-free", "antialias-rotated-semicircle-top-drawer-reset-free",
    "new-view-antialias-rotated-semicircle-top-drawer-reset-free",
    "new-view-antialias-rotated-semicircle-top-drawer-reset-free-large",
    "new-close-view-antialias-rotated-semicircle-top-drawer-reset-free",
]

dataset = "new-close-view-antialias-rotated-semicircle-top-drawer-reset-free"
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
# Rotated Drawer Reset-Free Data Reconstructed With VQ-VAE
elif dataset == "reconstructed-rotated-top-drawer-reset-free":
    VAL_DATA_PATH = "data/reconstructed_reset_free_v5_rotated_top_drawer/"
    demo_paths=[dict(path=VAL_DATA_PATH + 'reconstructed_reset_free_v5_rotated_top_drawer_demos_{}.pkl'.format(str(i)), obs_dict=True, is_demo=True, use_latents=True) for i in range(16)]
# Anti-aliased Rotated Drawer Reset-Free Data
elif dataset == "antialias-rotated-top-drawer-reset-free":
    VAL_DATA_PATH = "data/antialias_reset_free_v5_rotated_top_drawer/"
    demo_paths=[dict(path=VAL_DATA_PATH + 'antialias_reset_free_v5_rotated_top_drawer_demos_{}.pkl'.format(str(i)), obs_dict=True, is_demo=True, use_latents=True) for i in range(16)]
elif dataset == "antialias-right-top-drawer-reset-free":
    VAL_DATA_PATH = "data/antialias_reset_free_v5_right_top_drawer/"
    demo_paths=[dict(path=VAL_DATA_PATH + 'antialias_reset_free_v5_right_top_drawer_demos_{}.pkl'.format(str(i)), obs_dict=True, is_demo=True, use_latents=True) for i in range(16)]
elif dataset == "antialias-rotated-semicircle-top-drawer-reset-free":
    VAL_DATA_PATH = "data/antialias_reset_free_v5_rotated_semicircle_top_drawer/"
    demo_paths=[dict(path=VAL_DATA_PATH + 'antialias_reset_free_v5_rotated_semicircle_top_drawer_demos_{}.pkl'.format(str(i)), obs_dict=True, is_demo=True, use_latents=True) for i in range(16)]
elif dataset == "new-view-antialias-rotated-semicircle-top-drawer-reset-free":
    VAL_DATA_PATH = "data/new_view_antialias_reset_free_v5_rotated_semicircle_top_drawer/"
    demo_paths=[dict(path=VAL_DATA_PATH + 'new_view_antialias_reset_free_v5_rotated_semicircle_top_drawer_demos_{}.pkl'.format(str(i)), obs_dict=True, is_demo=True, use_latents=True) for i in range(8)]
elif dataset == "new-view-antialias-rotated-semicircle-top-drawer-reset-free-large":
    VAL_DATA_PATH = "data/new_view_antialias_reset_free_v5_rotated_semicircle_top_drawer_large/"
    demo_paths=[dict(path=VAL_DATA_PATH + 'new_view_antialias_reset_free_v5_rotated_semicircle_top_drawer_large_demos_{}.pkl'.format(str(i)), obs_dict=True, is_demo=True, use_latents=True) for i in range(32)]
elif dataset == "new-close-view-antialias-rotated-semicircle-top-drawer-reset-free":
    VAL_DATA_PATH = "data/new_close_view_antialias_reset_free_v5_rotated_semicircle_top_drawer/"
    demo_paths=[dict(path=VAL_DATA_PATH + 'new_close_view_antialias_reset_free_v5_rotated_semicircle_top_drawer_demos_{}.pkl'.format(str(i)), obs_dict=True, is_demo=True, use_latents=True) for i in range(32)]
else:
    assert False

vqvae = VAL_DATA_PATH + "best_vqvae_run29.pt" #VAL_DATA_PATH + "best_vqvae_run18_epoch3.pt" 
reward_classifier = VAL_DATA_PATH + "best_reward_classifier.pt"
pretrained_rl_path = VAL_DATA_PATH + "run155_id3_itr_-1.pt"
image_train_data = VAL_DATA_PATH + 'combined_images.npy'
image_test_data = VAL_DATA_PATH + 'combined_test_images.npy'

i = [0, 5, 10, 15]
j = [0, 5, 10]
reward_classifier_paths = [VAL_DATA_PATH + 'best_reward_classifier_{}_{}.pt'.format(i, j) for (i, j) in itertools.product(i, j)] 

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
        vf_kwargs=dict(
            hidden_sizes=[256, 256],
        ),

        trainer_kwargs=dict(
            discount=0.99,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,

            policy_weight_decay=0,
            q_weight_decay=0,

            reward_transform_kwargs=dict(m=1, b=-1),
            terminal_transform_kwargs=None,

            beta=0.1,
            quantile=0.9,
            clip_score=100,
        ),

        max_path_length=65, #50
        algo_kwargs=dict(
            batch_size=1024, #1024
            start_epoch=-150, # offline epochs
            num_epochs=1001, # online epochs
            num_eval_steps_per_epoch=1000,
            num_expl_steps_per_train_loop=1000,
            num_trains_per_train_loop=1000,
            num_online_trains_per_train_loop=1000,
            min_num_steps_before_training=4000,
            eval_epoch_freq=5,
            offline_expl_epoch_freq=5,
        ),
        replay_buffer_kwargs=dict(
            fraction_future_context=0.6,
            fraction_distribution_context=0.1,
            max_size=int(1E6),
        ),
        online_offline_split=True,
        online_offline_split_replay_buffer_kwargs=dict(
            online_replay_buffer_kwargs=dict(
                fraction_future_context=0.6,
                fraction_distribution_context=0.1,
                max_size=int(4E5),
            ),
            offline_replay_buffer_kwargs=dict(
                fraction_future_context=0.6,
                fraction_distribution_context=0.0,
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
            use_pretrained_reward_classifier_path=False,
            pretrained_reward_classifier_path=reward_classifier,
        ),

        observation_key='latent_observation',
        observation_keys=['latent_observation'],
        desired_goal_key='latent_desired_goal',
        save_video=True,
        expl_save_video_kwargs=dict(
            save_video_period=25,
            pad_color=0,
        ),
        eval_save_video_kwargs=dict(
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
            eval_goals_kwargs={},
            expl_goals='',
            expl_goals_kwargs={},
            training_goals='',
            training_goals_kwargs={},
        ),
        launcher_config=dict(
            unpack_variant=True,
            region='us-west-1', #HERE
        ),
        pretrained_rl_path=pretrained_rl_path,
    )

    search_space = {
        # Seed
        "seed": range(3),
        "env_kwargs.use_multiple_goals": [False],
        "eval_seeds": [1], # If 'use_multiple_goals'=False, use this evaluation environment seed
        "multiple_goals_eval_seeds": [[0, 1, 5, 7]], # If 'use_multiple_goals'=True, use list of evaluation environment seeds
        'env_type': ['top_drawer'],

        # Training Parameters
        "num_demos": [4], # Use first 'num_demos' demos for offline data
        "use_pretrained_rl_path": [False], # Load up existing policy/q-network/value network vs train a new one
        'algo_kwargs.start_epoch': [-100], # Negative epochs are pretraining. For only finetuning, set start_epoch=0.
        'trainer_kwargs.bc': [False], # Run BC experiment
        'algo_kwargs.num_online_trains_per_train_loop': [8000],
        "max_path_length": [100], # Length of trajectory during exploration and evaluation
        "algo_kwargs.num_expl_steps_per_train_loop": [1000], # Total number of steps during exploration per train loop
        'env_kwargs.drawer_sliding' : [True],
        'env_kwargs.reset_interval' : [1], # Reset environment every 'reset_interval' episodes

        ## Training Hyperparameters
        'trainer_kwargs.beta': [0.01], 

        'trainer_kwargs.use_online_beta': [False], # Overrides currently beta with beta_online during finetuning
        'trainer_kwargs.beta_online': [0.01],

        'trainer_kwargs.use_anneal_beta' : [False], # Anneal beta every 'anneal_beta_every' by 'anneal_beta_by until 'anneal_beta_stop_at'
        'trainer_kwargs.anneal_beta_every': [20],
        'trainer_kwargs.anneal_beta_by': [.05],
        'trainer_kwargs.anneal_beta_stop_at': [.0001],

        'reward_kwargs.use_pretrained_reward_classifier_path': [True], # If True, use pretrained reward classifier. If False, use epsilon.
        'reward_kwargs.pretrained_reward_classifier_path': [VAL_DATA_PATH + 'best_reward_classifier_5_0.pt'], #reward_classifier_paths,
        'reward_kwargs.reward_classifier_threshold': [0.95, 0.955, 0.96, 0.965, 0.97, 0.975, 0.98, 0.985, 0.99, 0.995],
        'reward_kwargs.epsilon': [3.0],
        'trainer_kwargs.quantile': [0.9],

        ## Network Parameters
        'gripper_observation' : [False], # Concatenate gripper position and rotation into network input
        "image": [False], # Latent-space or image-space
        "policy_kwargs.std": [0.15], # Fixed std of policy during exploration
        # "exploration_policy_kwargs.exploration_version": ['ou'],
        # "exploration_policy_kwargs.exploration_noise": [0.0, 0.1, 0.2, 0.3, 0.4],
        'qf_kwargs.output_activation': [Clamp(max=0)],

        ## Goals
        "use_both_ground_truth_and_affordance_expl_goals": [False],
        "affordance_sampling_prob": [1], # If "use_ground_truth_and_affordance_expl_goals"=True, this gives sampling proportion of affordance model during expl     
        "ground_truth_expl_goals": [False], # If ""use_ground_truth_and_affordance_expl_goals"=False, we use either PixelCNN expl goals or ground truth expl goals
        
        "only_not_done_goals": [True], # For ground truth goals, only select goals that are not achieved by the initialization
        'env_kwargs.full_open_close_init_and_goal' : [False],  # Initialize drawer to fully close or fully open. Alternative, initialized uniform random.
        'full_open_close_goal' : [False], # Only use ground truth goals that are near-fully open or closed.

        ## Relabeling
        "online_offline_split_replay_buffer_kwargs.online_replay_buffer_kwargs.fraction_distribution_context": [0.0],
        # "online_offline_split_replay_buffer_kwargs.online_replay_buffer_kwargs.fraction_future_context": [.8, 1.0],

        # "online_offline_split_replay_buffer_kwargs.online_replay_buffer_kwargs.preallocate_arrays": [True],
        # "online_offline_split_replay_buffer_kwargs.offline_replay_buffer_kwargs.preallocate_arrays": [True],
    }

    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    variants = []
    for variant in sweeper.iterate_hyperparameters():
        ## Error checking
        assert variant['algo_kwargs']['start_epoch'] % variant['algo_kwargs']['eval_epoch_freq'] == 0
        if variant['algo_kwargs']['start_epoch'] < 0:
            assert variant['algo_kwargs']['start_epoch'] % variant['algo_kwargs']['offline_expl_epoch_freq'] == 0
        if variant['use_pretrained_rl_path']:
            assert variant['algo_kwargs']['start_epoch'] == 0
        if variant['trainer_kwargs']['use_online_beta']:
            assert variant['trainer_kwargs']['use_anneal_beta'] == False
        if variant['multiple_goals_eval_seeds']:
            assert variant['only_not_done_goals'], 'multiple goals without filtering out bad goals not implemented yet'
            assert not variant['gripper_observation'], 'multiple goals with gripper observation not implemented yet'

        env_type = variant['env_type']
        if dataset != 'val' and env_type == 'pnp':
            env_type = 'obj'
        
        full_open_close_str = "full_open_close_" if variant['full_open_close_goal'] else ""
        if variant['env_kwargs']['use_multiple_goals']:
            eval_goals = []
            for eval_seed in variant['multiple_goals_eval_seeds']:
                eval_seed_str = f"_seed{eval_seed}"
                eval_goals.append(VAL_DATA_PATH + f'{full_open_close_str}{env_type}_goals{eval_seed_str}.pkl')
        else:
            eval_seed_str = f"_seed{variant['eval_seeds']}" if 'eval_seeds' in variant.keys() else ""
            eval_goals = VAL_DATA_PATH + f'{full_open_close_str}{env_type}_goals{eval_seed_str}.pkl'
        variant['presampled_goal_kwargs']['eval_goals'] = eval_goals

        variant['path_loader_kwargs']['demo_paths'] = variant['path_loader_kwargs']['demo_paths'][:variant['num_demos']]
        variant['online_offline_split_replay_buffer_kwargs']['offline_replay_buffer_kwargs']['max_size'] = min(int(6E5), int(500*75*variant['num_demos']))
        variant['online_offline_split_replay_buffer_kwargs']['online_replay_buffer_kwargs']['max_size'] = min(int(4/6 * 500*75*variant['num_demos']), int(1E6 - variant['online_offline_split_replay_buffer_kwargs']['offline_replay_buffer_kwargs']['max_size']))

        if variant['use_both_ground_truth_and_affordance_expl_goals']:
            variant['exploration_goal_sampling_mode']="conditional_vae_prior_and_not_done_presampled_images"
            variant['training_goal_sampling_mode']="presample_latents"
            variant['presampled_goal_kwargs']['expl_goals'] = eval_goals
            variant['presampled_goal_kwargs']['expl_goals_kwargs']['affordance_sampling_prob'] = variant['affordance_sampling_prob']
        elif variant['ground_truth_expl_goals']:
            variant['exploration_goal_sampling_mode']="presampled_images" #"presample_latents"
            variant['training_goal_sampling_mode']="presampled_images"
            variant['presampled_goal_kwargs']['expl_goals'] = eval_goals
            variant['presampled_goal_kwargs']['training_goals'] = eval_goals

        if variant['only_not_done_goals']:
            if variant['env_kwargs']['use_multiple_goals']:
                if variant["training_goal_sampling_mode"] == "presampled_images":
                    variant["training_goal_sampling_mode"] = "multiple_goals_not_done_presampled_images"
                if variant["exploration_goal_sampling_mode"] == "presampled_images":
                    variant["exploration_goal_sampling_mode"] = "multiple_goals_not_done_presampled_images"
                if variant["evaluation_goal_sampling_mode"] == "presampled_images":
                    variant["evaluation_goal_sampling_mode"] = "multiple_goals_not_done_presampled_images"
            else:
                if variant["training_goal_sampling_mode"] == "presampled_images":
                    variant["training_goal_sampling_mode"] = "not_done_presampled_images"
                if variant["exploration_goal_sampling_mode"] == "presampled_images":
                    variant["exploration_goal_sampling_mode"] = "not_done_presampled_images"
                if variant["evaluation_goal_sampling_mode"] == "presampled_images":
                    variant["evaluation_goal_sampling_mode"] = "not_done_presampled_images"

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
        elif dataset in ["rotated-top-drawer-reset-free", "reconstructed-rotated-top-drawer-reset-free"]:
            variant['env_class'] = SawyerRigAffordancesV1
        elif dataset in [
            "antialias-rotated-top-drawer-reset-free", "antialias-right-top-drawer-reset-free", 
            "antialias-rotated-semicircle-top-drawer-reset-free",
            "new-view-antialias-rotated-semicircle-top-drawer-reset-free",
            "new-view-antialias-rotated-semicircle-top-drawer-reset-free-large",
            "new-close-view-antialias-rotated-semicircle-top-drawer-reset-free",
            ]:
            variant['env_class'] = SawyerRigAffordancesV1
            variant['env_kwargs']['downsample'] = True
            variant['env_kwargs']['env_obs_img_dim'] = 196
            if dataset == "antialias-right-top-drawer-reset-free":
                variant['env_kwargs']['fix_drawer_orientation'] = True
            elif dataset == "antialias-rotated-semicircle-top-drawer-reset-free":
                variant['env_kwargs']['fix_drawer_orientation_semicircle'] = True
            elif dataset in ["new-view-antialias-rotated-semicircle-top-drawer-reset-free", "new-view-antialias-rotated-semicircle-top-drawer-reset-free-large"]:
                variant['env_kwargs']['fix_drawer_orientation_semicircle'] = True
                variant['env_kwargs']['new_view'] = True
            elif dataset == "new-close-view-antialias-rotated-semicircle-top-drawer-reset-free":
                variant['env_kwargs']['fix_drawer_orientation_semicircle'] = True
                variant['env_kwargs']['new_view'] = True
                variant['env_kwargs']['close_view'] = True
        else:
            assert False
        
        if 'eval_seeds' in variant.keys():
            variant['env_kwargs']['test_env_seed'] = variant['eval_seeds']
        if variant['env_kwargs']['use_multiple_goals']:
            variant['env_kwargs']['test_env_seeds'] = variant['multiple_goals_eval_seeds']
        
        if variant['gripper_observation']:
            variant['observation_keys'] = ['latent_observation', 'gripper_state_observation']

        # Image
        if variant['image']:
            assert 'gripper_observation' not in variant or not variant['gripper_observation'], "image-based not implemented yet"
            variant['policy_class'] = GaussianCNNPolicy
            variant['qf_class'] = ConcatCNN
            variant['vf_class'] = CNN
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
            variant['vf_kwargs'] = dict(
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
            
            variant['algo_kwargs']['batch_size'] = 256
            variant['replay_buffer_kwargs']['max_size'] = int(5E5)
        
        variants.append(variant)

    run_variants(iql_rig_experiment, variants, run_id=0, process_args_fn=process_args)