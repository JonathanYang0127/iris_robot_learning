import rlkit.util.hyperparameter as hyp
from rlkit.launchers.launcher_util import run_experiment
from rlkit.launchers.arglauncher import run_variants
from rlkit.torch.grill.common import train_reward_classifier

from rlkit.torch.networks.mlp import Mlp
from rlkit.torch.networks import Clamp, Sigmoid, SigmoidClamp

VAL_DATA_PATH = "data/new_close_view_antialias_reset_free_v5_rotated_semicircle_top_drawer/"

train_demo_paths = [VAL_DATA_PATH + 'new_close_view_antialias_reset_free_v5_rotated_semicircle_top_drawer_demos_{0}.pkl'.format(i) for i in range(0, 31)]
test_demo_paths = [VAL_DATA_PATH + 'new_close_view_antialias_reset_free_v5_rotated_semicircle_top_drawer_demos_{0}.pkl'.format(i) for i in range(31, 32)]
vqvae = '/2tb/home/patrickhaoy/data/affordances/experiments/patrick/val/train-vqvae/run29/id0/best_vqvae.pt'

def main():
    variant = dict(
        reward_classifier_class=Mlp,
        reward_classifier_kwargs=dict(
            hidden_sizes=[256, 256],
            output_size=1,
            input_size=1440,
        ),
        train_classifier_kwargs=dict(
            vqvae_path=vqvae,
            num_epochs=100,
            dataset_path={'train': train_demo_paths,
                          'test': test_demo_paths,
                          },
            batch_size=1024,
            state_indices=list(range(8, 11)),
            done_thres=0.065,
            cond_on_k_after_done=10, 
            positive_k_before_done=0,
            num_train_batches_per_epoch=100,
            num_test_batches_per_epoch=1,
        )
    )

    search_space = {
        "seed": range(1),
        "train_classifier_kwargs.cond_on_k_after_done": [10],
        "train_classifier_kwargs.positive_k_before_done": [0],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    variants = []
    for variant in sweeper.iterate_hyperparameters():
        variants.append(variant)

    run_variants(train_reward_classifier, variants, run_id=0) #HERE

if __name__ == "__main__":
    main()
