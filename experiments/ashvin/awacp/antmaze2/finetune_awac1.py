import rlkit.util.hyperparameter as hyp
from rlkit.launchers.arglauncher import run_variants

from rlkit.launchers.experiments.jaxrl.pr_launcher import pr_experiment, process_args

if __name__ == "__main__":
    variant = dict(
        env_name = "Ant-v2",
        dataset_name = "d4rl",
        # save_dir = "./tmp/",
        seedid = 0,
        eval_episodes = 10,
        log_interval = 1000,
        eval_interval = 10000,
        batch_size = 1024,
        max_steps = int(1e6),
        init_dataset_size = None,
        num_pretraining_steps = int(1e6),
        tqdm = True,
        save_video = False,
        config = dict(
            algo = 'awac',
            actor_optim_kwargs = dict(
                learning_rate = 3e-4,
                weight_decay = 1e-4,
            ),
            actor_hidden_dims = (256, 256, 256, 256),
            state_dependent_std = False,
            critic_lr = 3e-4,
            critic_hidden_dims = (256, 256),
            discount = 0.99,
            tau = 0.005,
            target_update_period = 1,
            beta = 10.0,
            num_samples = 1,
            replay_buffer_size = None,
        )
    )

    search_space = {
        'env_name': [
            "antmaze-umaze-v0", "antmaze-umaze-diverse-v0", "antmaze-medium-play-v0",
            "antmaze-medium-diverse-v0", "antmaze-large-diverse-v0", "antmaze-large-play-v0",
        ],
        'seedid': range(0, 3),
        'num_pretraining_steps': [int(50000), int(500000), int(1e6)],
    }

    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    variants = []
    for variant in sweeper.iterate_hyperparameters():
        variants.append(variant)

    run_variants(pr_experiment, variants, process_args)
