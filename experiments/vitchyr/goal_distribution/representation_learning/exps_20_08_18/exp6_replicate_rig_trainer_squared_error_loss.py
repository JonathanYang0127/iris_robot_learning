import rlkit.util.hyperparameter as hyp
from multiworld.envs.pygame import PickAndPlaceEnv
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.sets.vae_launcher import train_set_vae

if __name__ == "__main__":
    variant = dict(
        env_id='OneObject-PickAndPlace-BigBall-RandomInit-2D-v1',
        renderer_kwargs=dict(
            output_image_format='CHW',
        ),
        create_vae_kwargs=dict(
            latent_dim=128,
            encoder_cnn_kwargs=dict(
                kernel_sizes=[5, 3, 3],
                n_channels=[16, 32, 64],
                strides=[3, 2, 2],
                paddings=[0, 0, 0],
                pool_type='none',
                hidden_activation='relu',
            ),
            encoder_mlp_kwargs=dict(
                hidden_sizes=[],
            ),
            decoder_dcnn_kwargs=dict(
                kernel_sizes=[3, 3, 6],
                n_channels=[32, 16, 3],
                strides=[2, 2, 3],
                paddings=[0, 0, 0],
                normalization_type='batch',
            ),
            decoder_mlp_kwargs=dict(
                hidden_sizes=[],
                output_activation='sigmoid',
                normalization_type='batch',
            ),
            use_fancy_architecture=False,
            decoder_distribution="gaussian_fixed_unit_variance",
        ),
        vae_trainer_kwargs=dict(
            vae_lr=1e-3,
            vae_visualization_config=dict(
                num_recons=10,
                num_samples=20,
                # debug_period=50,
                debug_period=20,
                unnormalize_images=True,
                image_format='CHW',
            ),
            beta=1,
            set_loss_weight=0,
        ),
        data_loader_kwargs=dict(
            batch_size=128,
        ),
        vae_algo_kwargs=dict(
            num_iters=501,
            num_epochs_per_iter=1,
            progress_csv_file_name='vae_progress.csv',
        ),
        generate_test_set_kwargs=dict(
            num_samples_per_set=128,
            # create a new test set every run
            set_configs=[
                dict(
                    version='project_onto_axis',
                    axis_idx_to_value={
                        0: None,
                        1: None,
                    },
                ),
                dict(
                    version='project_onto_axis',
                    axis_idx_to_value={
                        0: None,
                    },
                ),
                dict(
                    version='project_onto_axis',
                    axis_idx_to_value={
                        2: None,
                        3: None,
                    },
                ),
                dict(
                    version='project_onto_axis',
                    axis_idx_to_value={
                        2: None,
                    },
                ),
            ],
        ),
        generate_train_set_kwargs=dict(
            num_sets=3,
            num_samples_per_set=128,
            saved_filename='/global/scratch/vitchyr/doodad-log-since-07-10-2020/manual-upload/sets/hand2xy_hand2x_1obj2xy_1obj2x_num_objs_1.pickle',
            # saved_filename='manual-upload/sets/hand2xy_hand2x_1obj2xy_1obj2x_num_objs_1.pickle',
        ),
        num_ungrouped_images=10000 - 3 * 128,
    )

    n_seeds = 1
    mode = 'local'
    exp_name = 'dev-{}'.format(
        __file__.replace('/', '-').replace('_', '-').split('.')[0]
    )

    n_seeds = 2
    mode = 'sss'
    exp_name = __file__.split('/')[-1].split('.')[0].replace('_', '-')
    print('exp_name', exp_name)

    search_space = {
        'create_vae_kwargs.use_fancy_architecture': [
            True, False,
        ],
        'vae_trainer_kwargs.beta': [
            1, 10, 50,
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    raise NotImplementedError("""
    To run this, replace log_prob computation in vae_torch_trainer with
    mean_log_prob = - ((p_x_given_z.mean - x)**2).sum()
    """)
    variants = list(sweeper.iterate_hyperparameters())
    for _ in range(n_seeds):
        for exp_id, variant in enumerate(variants):
            if mode == 'local':
                variant['vae_algo_kwargs']['num_iters'] = 1
                variant['generate_train_set_kwargs']['saved_filename'] = (
                    'manual-upload/sets/hand2xy_hand2x_1obj2xy_1obj2x_num_objs_1.pickle'
                )
            run_experiment(
                train_set_vae,
                exp_name=exp_name,
                num_exps_per_instance=2,
                mode=mode,
                variant=variant,
                # slurm_config_name='cpu',
                use_gpu=True,
                # gpu_id=1,
            )
