import rlkit.util.hyperparameter as hyp
from rlkit.demos.source.encoder_dict_to_mdp_path_loader import EncoderDictToMDPPathLoader
from rlkit.launchers.experiments.ashvin.awac_rig import awac_rig_experiment
from rlkit.launchers.launcher_util import run_experiment
from rlkit.launchers.arglauncher import run_variants
from rlkit.torch.sac.policies import GaussianPolicy, GaussianMixturePolicy
from roboverse.envs.sawyer_rig_multiobj_v0 import SawyerRigMultiobjV0
from roboverse.envs.sawyer_rig_multiobj_tray_v0 import SawyerRigMultiobjTrayV0
from roboverse.envs.sawyer_rig_affordances_v0 import SawyerRigAffordancesV0
from rlkit.torch.networks import Clamp
from rlkit.torch.vae.vq_vae import VQ_VAE
from rlkit.torch.vae.vq_vae_trainer import VQ_VAETrainer
from rlkit.torch.grill.common import train_vqvae

VAL_DATA_PATH = "sasha/affordances/combined/"

image_train_data = VAL_DATA_PATH + 'combined_images.npy'
image_test_data = VAL_DATA_PATH + 'combined_test_images.npy'

def process_args(variant):
    if variant.get("debug", False):
        variant.update(dict(
            num_epochs=1,
            algo_kwargs=dict(
                batch_size=2,
                key_to_reconstruct='x_t',
            ),
            train_pixelcnn_kwargs=dict(
                num_epochs=1,
                data_size=10,
                num_train_batches_per_epoch=2,
                num_test_batches_per_epoch=2,
                dump_samples=False,
            ),
        ))

def main():
    variant = dict(
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
        launcher_config=dict(
            unpack_variant=False,
            region='us-west-1', #HERE
        ),
    )

    search_space = {
        "seed": range(2),
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    variants = []
    for variant in sweeper.iterate_hyperparameters():
        variants.append(variant)

    run_variants(train_vqvae, variants, run_id=0, process_args_fn=process_args) #HERE

if __name__ == "__main__":
    main()
