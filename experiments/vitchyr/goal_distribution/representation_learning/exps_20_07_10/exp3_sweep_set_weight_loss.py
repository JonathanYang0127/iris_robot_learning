import rlkit.util.hyperparameter as hyp

from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.sets.vae_launcher import train_set_vae

if __name__ == '__main__':
    variant = dict(
        latent_dim=128,
        create_vae_kwargs=dict(
            encoder_cnn_kwargs=dict(
                kernel_sizes=[4],
                n_channels=[128],
                strides=[2],
                paddings=[0],

                # kernel_sizes=[3, 3, 3],
                # n_channels=[32, 64, 128],
                # strides=[1, 1, 1],
                # paddings=[0, 0, 0],

                pool_type='none',
                hidden_activation='relu',
            ),
            encoder_mlp_kwargs=dict(
                # hidden_sizes=[128, 128],
                hidden_sizes=[],
            ),
            decoder_mlp_kwargs=dict(
                hidden_sizes=[256, 256],
            ),
            decoder_distribution='bernoulli',
            use_mlp_decoder=True,
        ),
        vae_trainer_kwargs=dict(
            vae_lr=1e-3,
            vae_visualization_config=dict(
                num_recons=5,
                num_samples=20,
                debug_period=50,
                # debug_period=10,
                unnormalize_images=True,
            ),
            # beta=1,
            # set_loss_weight=1,
            beta=0.001,
            set_loss_weight=0,
        ),
        data_loader_kwargs=dict(
            batch_size=32,
        ),
        algo_kwargs=dict(
            num_iters=1001,
            num_epochs_per_iter=10,
            # num_epochs=101,
            # num_epochs=11,
        ),
        generate_set_kwargs=dict(
            num_sets=5,
            num_samples_per_set=32,
        ),
        num_ungrouped_images=64
    )
    n_seeds = 1

    search_space = {
        'vae_trainer_kwargs.beta': [
            # 0.01, 0.1, 1, 10, 100,
            0.1, 1,
        ],
        'vae_trainer_kwargs.set_loss_weight': [
            0, 0.01, 1, 100,
        ],
        'create_vae_kwargs.decoder_distribution': [
            'gaussian_learned_global_image_variance',
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        beta = variant['vae_trainer_kwargs']['beta']
        slw = variant['vae_trainer_kwargs']['set_loss_weight']
        for _ in range(n_seeds):
            variant['logger_config'] = dict(
                trial_dir_suffix='beta-{}-slw{}'.format(
                    beta,
                    slw,
                )
            )
            run_experiment(
                train_set_vae,
                variant=variant,
                exp_name='vae-encoder-set-loss-sweep',
                mode='sss',
                # exp_name='dev-vae-encoder-sweep',
                # mode='here_no_doodad',
                # slurm_config_name='gpu_fc',
                # slurm_config_name='gpu_low_pri',
                # exp_name='vae-bernoulli-decoder',
                use_gpu=True,
            )
