import rlkit.util.hyperparameter as hyp
from rlkit.launchers.arglauncher import run_variants

from rlkit.launchers.experiments.jaxrl.pr_launcher import pr_experiment

if __name__ == "__main__":
    variant = dict(
        env_name = "Ant-v2",
        dataset_name = "awac",
        # save_dir = "./tmp/",
        seedid = 0,
        eval_episodes = 10,
        log_interval = 1000,
        eval_interval = 10000,
        batch_size = 1024,
        max_steps = int(1e6),
        init_dataset_size = None,
        num_pretraining_steps = int(5e4),
        tqdm = True,
        save_video = False,
        config = dict(
            algo = 'pr',
            actor_lr = 3e-4,
            value_lr = 3e-4,
            critic_lr = 3e-4,
            hidden_dims = (256, 256),
            discount = 0.99,
            quantile = 0.8,
            temperature = 10.0,
            tau = 0.005,
            target_update_period = 1,
            replay_buffer_size = None,
            q_kwargs = dict(
                clamp = dict(a_max=2,),
            ),
            adv_clip=2,
            policy_weight_decay = 1e-4,
        ),
        awac_activation="softmax",
        action_noise_scale=0.1,
    )

    search_space = {
        'env_name': ["relocate-binary-v0", "door-binary-v0", "pen-binary-v0",],
        'seedid': range(3, 6),
        'config.quantile': [0.5, 0.8, 0.9],
        'config.temperature': [1.0, 3.0, 10.0],
        'action_noise_scale': [0.03, ],
        'config.policy_weight_decay': [1e-4],
    }

    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    variants = []
    for variant in sweeper.iterate_hyperparameters():
        variants.append(variant)

    run_variants(pr_experiment, variants, )
