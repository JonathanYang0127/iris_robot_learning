from __future__ import print_function
import rlkit.util.hyperparameter as hyp
from rlkit.launchers.arglauncher import run_variants
from rlkit.torch.gan.bigan import BiGAN
from rlkit.torch.gan.bigan_trainer import BiGANTrainer
from multiworld.envs.pygame.multiobject_pygame_env import Multiobj2DEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_multiobj_subset import SawyerMultiobjectEnv
from rlkit.launchers.config import CIFAR10_DATASET
from rlkit.launchers.experiments.danieljing.gan_launcher import train_gan

if __name__ == "__main__":

    variant = dict(
        num_epochs=20,
        dataset = 'cifar10',
        dataroot = CIFAR10_DATASET,
        num_workers = 2,
        batch_size = 100,
        image_size = 32,
        gan_trainer_class=BiGANTrainer,
        gan_class=BiGAN,
        ngpu = 1,
        beta = 0.5,
        lr = 1e-4,
        latent_size = 256,
        dropout = 0.2,
        output_size = 1,
        generator_threshold = 3.5,
        #nc = 3,
        #ngf =
        #ndf =

        save_period=25,
        logger_variant=dict(
            tensorboard=True,
        ),

        slurm_variant=dict(
            timeout_min=48 * 60,
            cpus_per_task=10,
            gpus_per_node=1,
        ),
    )
    search_space = {
        'seedid': range(1)
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    variants = []
    for variant in sweeper.iterate_hyperparameters():
        variants.append(variant)

    run_variants(train_gan, variants, run_id=0)
