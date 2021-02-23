import rlkit.misc.hyperparameter as hyp
from experiments.murtaza.multiworld.skew_fit.reacher.generate_uniform_dataset import generate_uniform_dataset_reacher
from multiworld.envs.mujoco.cameras import sawyer_init_camera_zoomed_in, sawyer_pusher_camera_upright_v2
from rlkit.launchers.launcher_util import run_experiment
import rlkit.torch.vae.vae_schedules as vae_schedules
from rlkit.torch.vae.conv_vae import imsize48_default_architecture, imsize48_default_architecture_with_more_hidden_layers
from rlkit.launchers.arglauncher import run_variants
from rlkit.torch.grill.cvae_experiments import (
    grill_her_td3_offpolicy_online_vae_full_experiment,
)
from rlkit.torch.grill.common import train_vae
from rlkit.misc.ml_util import PiecewiseLinearSchedule, ConstantSchedule
from multiworld.envs.pygame.multiobject_pygame_env import Multiobj2DEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_multiobj_subset import SawyerMultiobjectEnv
from rlkit.torch.vae.vq_vae import VQ_VAE
from rlkit.torch.vae.vq_vae_trainer import VQ_VAETrainer
from rlkit.data_management.online_conditional_vae_replay_buffer import \
        OnlineConditionalVaeRelabelingBuffer

if __name__ == "__main__":
    variant = dict(
        beta=1,
        imsize=48,
        num_epochs=1501,
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
            #num_hiddens=256,
            #num_residual_layers=4,
            #num_residual_hiddens=128,
        ),

        algo_kwargs=dict(
            key_to_reconstruct='x_t',
            start_skew_epoch=5000,
            is_auto_encoder=False,
            batch_size=128,
            lr=1e-3, #1E-4
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

        launcher_config=dict(
            #unpack_variant=True,
            region='us-west-2',
        ),
    ),

    search_space = {
        'seed': range(1), #2
        'embedding_dim': [2,],
        'vae_kwargs.decay': [0],#[0, 0.99],
        'vae_kwargs.num_embeddings':[256],#[256, 512],
        'vae_kwargs.num_residual_layers': [2], #[2, 3],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    variants = []
    for variant in sweeper.iterate_hyperparameters():
        variants.append(variant)

    run_variants(train_vae, variants, run_id=1)