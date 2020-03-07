"""
AWR + SAC from demo experiment
"""

from railrl.demos.source.dict_to_mdp_path_loader import DictToMDPPathLoader
from railrl.launchers.experiments.ashvin.awr_sac_rl import experiment

import railrl.misc.hyperparameter as hyp
from railrl.launchers.arglauncher import run_variants

if __name__ == "__main__":
    variant = dict(
        num_epochs=1,
        num_eval_steps_per_epoch=10000,
        num_trains_per_train_loop=1000,
        num_expl_steps_per_train_loop=10000,
        min_num_steps_before_training=1000,
        max_path_length=1000,
        batch_size=256,
        replay_buffer_size=int(1E6),
        algorithm="SAC",
        version="normal",
        collection_mode='batch',

        layer_size=256,
        policy_kwargs=dict(
            hidden_sizes=[256, 256, 256, 256],
        ),

        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            beta=1,
            use_automatic_entropy_tuning=True,

            bc_num_pretrain_steps=50000,
            # q_num_pretrain_steps=0,
            policy_weight_decay=1e-4,
            bc_loss_type="mle",
        ),
        num_exps_per_instance=1,
        region='us-west-2',

        path_loader_class=DictToMDPPathLoader,
        path_loader_kwargs=dict(
            obs_key="state_observation",
            demo_paths=[
                dict(
                    path="demos/icml2020/hand/door.npy",
                    obs_dict=True,
                    is_demo=True,
                ),
            ],
        ),

        logger_variant=dict(
            tensorboard=True,
        ),
        load_demos=True,
        pretrain_policy=True,
        pretrain_rl=True,

        save_paths=True,
    )

    search_space = {
        'env': ["door-v0", ],
        'seedid': range(10),
        'trainer_kwargs.beta': [10, ],
    }

    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    variants = []
    for variant in sweeper.iterate_hyperparameters():
        variants.append(variant)

    run_variants(experiment, variants, run_id=0)
