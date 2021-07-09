import rlkit.util.hyperparameter as hyp
from rlkit.demos.source.dict_to_mdp_path_loader import EncoderDictToMDPPathLoader
from rlkit.launchers.experiments.ashvin.awac_rig import awac_rig_experiment
from rlkit.launchers.launcher_util import run_experiment
from rlkit.launchers.arglauncher import run_variants
from rlkit.torch.sac.policies import GaussianPolicy, GaussianMixturePolicy
from roboverse.envs.sawyer_rig_multiobj_v0 import SawyerRigMultiobjV0
from rlkit.torch.networks import Clamp
from rlkit.torch.vae.vq_vae import VQ_VAE
from rlkit.torch.vae.vq_vae_trainer import VQ_VAETrainer
from rlkit.torch.grill.common import train_vqvae

complete_object_list = ['l_sofa', 'conic_bin', 'square_prism_bin', 'bunsen_burner', 'wide_circular_vase', 'flat_circular_basket', 'pitcher','gatorade', 'narrow_tray', 
'jar', 'horn_vase', 'fountain_vase', 'narrow_top_vase', 'long_vase', 'conic_cup', 'conic_bowl','ball', 'hex_deep_bowl', 'square_deep_bowl',
'pillow', 'shed', 'oblong_scooper', 'sack_vase', 'smushed_dumbbell', 'grill_bench','circular_picnic_table', 'circular_table', 'short_handle_cup',
'curved_handle_cup', 'flowery_half_donut', 'semi_golf_ball_bowl', 'passenger_airplane','open_top_rect_box', 'cookie_circular_lidless_tin', 'chipotle_bowl',
'toilet_bowl', 'buffet_food_tray', 'keyhole', 'bathtub', 'crooked_rim_capsule_container','colunnade_top', 'pacifier_vase', 'square_rod_embellishment', 'bongo_drum_bowl',
'flat_bottom_sack_vase', 'stalagcite_chunk', 'pear_ringed_vase', 'two_handled_vase','goblet', 'ringed_cup_oversized_base', 't_cup', 'teepee', 'bullet_vase',
'haystack_sofa', 'box_wood_frame', 'rect_spotted_hollow_bottom_sofa', 'box_sofa','earmuff', 'l_automatic_faucet', 'double_l_faucet', 'box_crank', 'glass_half_gallon',
'pepsi_bottle', 'two_layered_lampshade', 'beehive_funnel', 'rabbit_lamp','elliptical_capsule', 'trapezoidal_bin', 'staple_table', 'grill_park_bench', 'thick_wood_chair',
'park_grill_chair', 'long_half_pipe_smooth_park_chair', 'flat_boat_dish','modern_canoe', 'vintage_canoe', 'oil_tanker', 'x_curved_modern_bookshelf', 'pitchfork_shelf',
'baseball_cap', 'tongue_chair', 'crooked_lid_trash_can','aero_cylinder']

demo_paths_1=[dict(path='sasha/complex_obj/gr_train_complex_obj_demos_0.pkl', obs_dict=True, is_demo=True),
                dict(path='sasha/complex_obj/gr_train_complex_obj_demos_1.pkl', obs_dict=True, is_demo=True),
                dict(path='sasha/complex_obj/gr_train_complex_obj_demos_2.pkl', obs_dict=True, is_demo=True)]

demo_paths_2=[dict(path='sasha/complex_obj/gr_train_complex_obj_demos_0.pkl', obs_dict=True, is_demo=True),
             dict(path='sasha/complex_obj/gr_train_complex_obj_demos_1.pkl', obs_dict=True, is_demo=True)]

demo_paths_3=[dict(path='sasha/complex_obj/gr_train_complex_obj_demos_0.pkl', obs_dict=True, is_demo=True)]

demo_paths_4=[dict(path='sasha/complex_obj/gr_train_complex_obj_demos_0.pkl',obs_dict=True, is_demo=True, data_split=0.5,)]

demo_paths_5=[dict(path='sasha/complex_obj/gr_train_complex_obj_demos_0.pkl',obs_dict=True, is_demo=True, data_split=0.25,)]


beer_bottle_goals = 'sasha/presampled_goals/3dof_beer_bottle_presampled_goals.pkl'
camera_goals = 'sasha/presampled_goals/3dof_camera_presampled_goals.pkl'
grill_trash_can_goals = 'sasha/presampled_goals/3dof_grill_trash_can_presampled_goals.pkl'
long_sofa_goals = 'sasha/presampled_goals/3dof_long_sofa_presampled_goals.pkl'
mug_goals = 'sasha/presampled_goals/3dof_mug_presampled_goals.pkl'

quat_dict={'mug': [0, -1, 0, 1],
        'long_sofa': [0, 0, 0, 1],
        'camera': [-1, 0, 0, 0],
        'grill_trash_can': [0, 0, 1, 1],
        'beer_bottle': [0, 0, 1, -1]}

if __name__ == "__main__":
    variant = dict(
        imsize=48,
        env_class=SawyerRigMultiobjV0,
        env_kwargs=dict(
            DoF=3,
            random_color_p=0,
            quat_dict=quat_dict,
            randomize=False,
            use_bounding_box=True,
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

            bc_num_pretrain_steps=0,
            q_num_pretrain1_steps=0,
            q_num_pretrain2_steps=25000, #25000
            policy_weight_decay=1e-4,
            q_weight_decay=0,

            rl_weight=1.0,
            use_awr_update=True,
            use_reparam_update=False,
            compute_bc=True,
            reparam_weight=0.0,
            awr_weight=1.0,
            bc_weight=0.0,

            reward_transform_kwargs=None,
            terminal_transform_kwargs=None,
        ),

        max_path_length=50, #50
        algo_kwargs=dict(
            batch_size=1024, #1024
            num_epochs=1001, #1001
            num_eval_steps_per_epoch=1000, #1000
            num_expl_steps_per_train_loop=1000, #1000
            num_trains_per_train_loop=1000, #1000
            min_num_steps_before_training=4000, #4000
        ),
        replay_buffer_kwargs=dict(
            #fraction_next_context=0.0,
            fraction_future_context=0.6,
            fraction_distribution_context=0.1,
            max_size=int(2E5),
        ),
        demo_replay_buffer_kwargs=dict(
            #fraction_next_context=0.0,
            fraction_future_context=0.6,
            fraction_distribution_context=0.1,
        ),
        reward_kwargs=dict(
            reward_type='sparse',
            epsilon=1.0,
        ),

        observation_key='latent_observation',
        desired_goal_key='latent_desired_goal',
        save_video=True,
        save_video_kwargs=dict(
            save_video_period=25,
            pad_color=0,
        ),

        reset_keys_map=dict(
            image_observation="initial_latent_state"
        ),

        path_loader_class=EncoderDictToMDPPathLoader,
        path_loader_kwargs=dict(
            recompute_reward=True,
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
        pretrain_policy=True,
        pretrain_rl=True,

        evaluation_goal_sampling_mode="presampled_images",
        exploration_goal_sampling_mode="presample_latents",

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
                dataset_path={'train': 'sasha/complex_obj/gr_train_complex_obj_images.npy',
                          'test': 'sasha/complex_obj/gr_test_complex_obj_images.npy'},
                augment_data=False,
                use_cached=False,
                show=False,
                oracle_dataset=False,
                oracle_dataset_using_set_to_goal=False,
                non_presampled_goal_img_is_garbage=False,
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

            save_period=10,
        ),
        train_model_func=train_vqvae,
        presampled_goal_kwargs=dict(
            eval_goals=grill_trash_can_goals, # HERE
            expl_goals='',
        ),
        launcher_config=dict(
            unpack_variant=True,
            region='us-west-1', #HERE
        ),
        #num_presample=50,
    )

    search_space = {
        "seed": range(2),
        'path_loader_kwargs.demo_paths': [demo_paths_1],
        'env_kwargs.object_subset': [['grill_trash_can']], #HERE

        'reward_kwargs.epsilon': [5.5],
        'trainer_kwargs.beta': [0.3,],
        'num_pybullet_objects':[5, 10, 15, 20, 25, 30, 35, 40],

        'policy_kwargs.min_log_std': [-6],
        'trainer_kwargs.awr_weight': [1.0],
        'trainer_kwargs.awr_use_mle_for_vf': [True, ],
        'trainer_kwargs.awr_sample_actions': [False, ],
        'trainer_kwargs.clip_score': [2, ],
        'trainer_kwargs.awr_min_q': [True, ],
        'trainer_kwargs.reward_transform_kwargs': [None, ],
        'trainer_kwargs.terminal_transform_kwargs': [dict(m=0, b=0),],
        'qf_kwargs.output_activation': [Clamp(max=0)],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    variants = []
    for variant in sweeper.iterate_hyperparameters():

        num_obj = variant['num_pybullet_objects']
        del variant['num_pybullet_objects']
        
        if num_obj is not None:
            object_list = complete_object_list[:num_obj]

            def demo_filter(path):
                return path["object_name"] in object_list

            def img_filter(dataset):
                keep_ind = 0
                objects_np = np.array(dataset['object'])
                for o in object_list:
                    keep_ind += (objects_np == o)

                keep_ind = np.where(keep_ind == 1)
                dataset['env'] = dataset['env'][keep_ind]
                dataset['observations'] = dataset['observations'][keep_ind]
                return dataset

            variant['path_loader_kwargs']['data_filter_fn'] = demo_filter
            variant['train_vae_kwargs']['generate_vae_dataset_kwargs']['data_filter_fn'] = img_filter
        
        variants.append(variant)

    run_variants(awac_rig_experiment, variants, run_id=12) #HERE