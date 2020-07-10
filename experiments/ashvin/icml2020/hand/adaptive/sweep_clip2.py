"""
AWR + SAC from demo experiment
"""

from railrl.demos.source.dict_to_mdp_path_loader import DictToMDPPathLoader
from railrl.launchers.experiments.awac.awac_rl import experiment, process_args

import railrl.misc.hyperparameter as hyp
from railrl.launchers.arglauncher import run_variants

from railrl.torch.sac.policies import GaussianPolicy
from railrl.torch.networks import Clamp

if __name__ == "__main__":
    variant = dict(
        num_epochs=1001,
        num_eval_steps_per_epoch=1000,
        num_trains_per_train_loop=1000,
        num_expl_steps_per_train_loop=1000,
        min_num_steps_before_training=1000,
        max_path_length=1000,
        batch_size=1024,
        replay_buffer_size=int(1E6),

        layer_size=256,
        policy_class=GaussianPolicy,
        policy_kwargs=dict(
            hidden_sizes=[256, 256, 256, 256],
            max_log_std=0,
            min_log_std=-6,
            std_architecture="values",
            # num_gaussians=1,
        ),

        algorithm="SAC",
        version="normal",
        collection_mode='batch',
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
            q_num_pretrain2_steps=25000,
            policy_weight_decay=1e-4,
            q_weight_decay=0,
            bc_loss_type="mse",

            rl_weight=1.0,
            use_awr_update=True,
            use_reparam_update=False,
            reparam_weight=0.0,
            awr_weight=0.0,
            bc_weight=1.0,

            post_bc_pretrain_hyperparams=dict(
                bc_weight=0.0,
                compute_bc=False,
            ),

            reward_transform_kwargs=None, # r' = r + 1
            terminal_transform_kwargs=None, # t = 0
            clip_score=None,
            validation_qlearning=True,
        ),
        launcher_config=dict(
            num_exps_per_instance=1,
            region='us-west-2',
        ),

        path_loader_class=DictToMDPPathLoader,
        path_loader_kwargs=dict(
            obs_key="state_observation",
            demo_paths=[
                # dict(
                #     path="demos/icml2020/hand/pen2_sparse.npy",
                #     obs_dict=True,
                #     is_demo=True,
                # ),
                # dict(
                #     path="demos/icml2020/hand/pen_bc5.npy",
                #     obs_dict=False,
                #     is_demo=False,
                #     train_split=0.9,
                # ),
            ],
        ),
        add_env_demos=True,
        add_env_offpolicy_data=True,

        # logger_variant=dict(
        #     tensorboard=True,
        # ),
        load_demos=True,
        pretrain_policy=True,
        pretrain_rl=True,
        # save_pretrained_algorithm=True,
        # snapshot_mode="all",
        use_validation_buffer=True,
    )

    search_space = {
        'env': ["pen-sparse-v0", "door-sparse-v0", "relocate-sparse-v0", ],
        # 'env': ["relocate-sparse-v0", ],
        'trainer_kwargs.bc_loss_type': ["mle"],
        'trainer_kwargs.awr_loss_type': ["mle"],
        'seedid': range(3),
        'trainer_kwargs.beta': [0.3, 0.6, ],

        'trainer_kwargs.clip_score': [3, ],

        'trainer_kwargs.reparam_weight': [0.0, ],
        'trainer_kwargs.awr_weight': [1.0],
        'trainer_kwargs.bc_weight': [1.0, ],
        # 'trainer_kwargs.q_update_period': [1, 2],
        # 'trainer_kwargs.policy_update_period': [1, 2],
        'policy_kwargs.std_architecture': ["values", ],

        # 'trainer_kwargs.compute_bc': [True, ],
        'trainer_kwargs.awr_use_mle_for_vf': [True, ],
        'trainer_kwargs.awr_sample_actions': [False, ],
        'trainer_kwargs.awr_min_q': [True, ],

        'trainer_kwargs.q_weight_decay': [0, 1e-5, ],

        'trainer_kwargs.reward_transform_kwargs': [None, ],
        'trainer_kwargs.terminal_transform_kwargs': [dict(m=0, b=0), ],
        'qf_kwargs.output_activation': [Clamp(max=0)],
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
