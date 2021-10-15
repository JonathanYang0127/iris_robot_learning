"""
AWR + SAC from demo experiment
"""

from rlkit.demos.source.hdf5_path_loader import HDF5PathLoader
from rlkit.launchers.experiments.awac.awac_rl import experiment, process_args

import rlkit.util.hyperparameter as hyp
from rlkit.launchers.arglauncher import run_variants

from rlkit.torch.sac.policies import GaussianPolicy
from rlkit.torch.sac.iql_trainer import IQLTrainer

# import d4rl.gym_mujoco
import d4rl

def main():
    variant = dict(
        algo_kwargs=dict(
            start_epoch=-1000, # offline epochs
            num_epochs=1001, # online epochs
            batch_size=256,
            num_eval_steps_per_epoch=1000,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
        ),
        max_path_length=1000,
        replay_buffer_size=int(2E6),
        layer_size=256,
        policy_class=GaussianPolicy,
        policy_kwargs=dict(
            hidden_sizes=[256, 256, ],
            max_log_std=0,
            min_log_std=-6,
            std_architecture="values",
            # num_gaussians=1,
        ),
        qf_kwargs=dict(
            hidden_sizes=[256, 256, ],
        ),

        algorithm="SAC",
        version="normal",
        collection_mode='batch',
        trainer_class=IQLTrainer,
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
            compute_bc=False,

            bc_num_pretrain_steps=0,
            q_num_pretrain1_steps=0,
            q_num_pretrain2_steps=0, # changed this line
            policy_weight_decay=1e-4,
            q_weight_decay=0,
            # bc_loss_type="mse",

            rl_weight=1.0,
            use_awr_update=True,
            use_reparam_update=False,
            reparam_weight=0.0,
            awr_weight=1.0,
            bc_weight=0.0,

            reward_transform_kwargs=None, # r' = r + 1
            terminal_transform_kwargs=None, # t = 0
            validation_qlearning=False, # changed this line, added
        ),
        launcher_config=dict(
            num_exps_per_instance=1,
            region='us-west-2',
        ),

        path_loader_class=HDF5PathLoader,
        path_loader_kwargs=dict(),
        add_env_demos=False,
        add_env_offpolicy_data=False,

        # logger_variant=dict(
        #     tensorboard=True,
        # ),
        load_demos=False,
        load_env_dataset_demos=True,
        pretrain_policy=True,
        pretrain_rl=True,
        # save_pretrained_algorithm=True,
        # snapshot_mode="all",
    )

    search_space = {
        'normalize_env': [False],
        'trainer_kwargs.normalize_over_state': ['advantage', ],
        'use_validation_buffer': [False], # changed this line, added
        'policy_kwargs.std': [None, ],
        'env_id': [
            "antmaze-umaze-v0", "antmaze-umaze-diverse-v0", "antmaze-medium-play-v0",
            "antmaze-medium-diverse-v0", "antmaze-large-diverse-v0", "antmaze-large-play-v0",
        ],
        'seedid': range(1),
        'trainer_kwargs.beta': [0.1, 1, 10],

        'policy_kwargs.std_architecture': ["values", ],
        'trainer_kwargs.awr_use_mle_for_vf': [True, ],
        'trainer_kwargs.awr_sample_actions': [False, ],
        'trainer_kwargs.awr_min_q': [True, ],

        'trainer_kwargs.q_weight_decay': [0, ],

        'trainer_kwargs.reward_transform_kwargs': [dict(m=1, b=-1), ],
        # 'trainer_kwargs.terminal_transform_kwargs': [dict(m=0, b=0), ],
        # 'qf_kwargs.output_activation': [Clamp(max=0)],
        # 'trainer_kwargs.train_bc_on_rl_buffer':[True],
        # 'policy_kwargs.num_gaussians': [1, ],
    }

    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    variants = []
    for variant in sweeper.iterate_hyperparameters():
        variants.append(variant)

    run_variants(experiment, variants, process_args)

if __name__ == "__main__":
    main()
