import rlkit.util.hyperparameter as hyp
from rlkit.launchers.launcher_util import run_experiment
from rlkit.launchers.arglauncher import run_variants

from roboverse.envs.sawyer_rig_affordances_v0 import SawyerRigAffordancesV0

from rlkit.demos.source.hdf5_path_loader import HDF5PathLoader
from rlkit.launchers.experiments.awac.finetune_image_rl import experiment, process_args

from rlkit.launchers.launcher_util import run_experiment

from rlkit.torch.sac.policies import GaussianPolicy
from rlkit.torch.sac.iql_trainer import IQLTrainer

from rlkit.demos.source.encoder_dict_to_mdp_path_loader import EncoderDictToMDPPathLoader
from rlkit.demos.source.dict_to_mdp_path_loader import DictToMDPPathLoader

from rlkit.torch.sac.policies import GaussianPolicy, GaussianMixturePolicy, GaussianCNNPolicy
from rlkit.torch.networks.cnn import ConcatCNN, CNN
from rlkit.torch.networks import ConcatMlp, Mlp

import random

# VAL_DATA_PATH = "sasha/affordances/combined/"
VAL_DATA_PATH = "/media/ashvin/s3doodad/sasha/affordances/combined/"

demo_paths=[dict(path=VAL_DATA_PATH + 'drawer_demos_0.pkl', obs_dict=True, is_demo=True),
            # dict(path=VAL_DATA_PATH + 'drawer_demos_1.pkl', obs_dict=True, is_demo=True),
            # dict(path=VAL_DATA_PATH + 'pnp_demos_0.pkl', obs_dict=True, is_demo=True),
            # dict(path=VAL_DATA_PATH + 'tray_demos_0.pkl', obs_dict=True, is_demo=True),

            # dict(path=VAL_DATA_PATH + 'drawer_demos_2.pkl', obs_dict=True, is_demo=True),
            # dict(path=VAL_DATA_PATH + 'drawer_demos_3.pkl', obs_dict=True, is_demo=True),
            # dict(path=VAL_DATA_PATH + 'pnp_demos_1.pkl', obs_dict=True, is_demo=True),
            # dict(path=VAL_DATA_PATH + 'tray_demos_1.pkl', obs_dict=True, is_demo=True),

            # dict(path=VAL_DATA_PATH + 'drawer_demos_4.pkl', obs_dict=True, is_demo=True),
            # dict(path=VAL_DATA_PATH + 'drawer_demos_5.pkl', obs_dict=True, is_demo=True),
            # dict(path=VAL_DATA_PATH + 'pnp_demos_2.pkl', obs_dict=True, is_demo=True),
            # dict(path=VAL_DATA_PATH + 'tray_demos_2.pkl', obs_dict=True, is_demo=True),

            # dict(path=VAL_DATA_PATH + 'drawer_demos_6.pkl', obs_dict=True, is_demo=True),
            # dict(path=VAL_DATA_PATH + 'drawer_demos_7.pkl', obs_dict=True, is_demo=True),
            # dict(path=VAL_DATA_PATH + 'pnp_demos_3.pkl', obs_dict=True, is_demo=True),
            # dict(path=VAL_DATA_PATH + 'tray_demos_3.pkl', obs_dict=True, is_demo=True),
            ]

if __name__ == "__main__":
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
        replay_buffer_size=int(2E5),
        layer_size=256,
        policy_class=GaussianPolicy,
        policy_kwargs=dict(
            hidden_sizes=[256, 256, ],
            max_log_std=0,
            min_log_std=-6,
            std_architecture="values",
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
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            soft_target_tau=0.005,

            policy_weight_decay=0,
            q_weight_decay=0,

            reward_transform_kwargs=None,
            terminal_transform_kwargs=None,

            beta=1.0 / 3,
            quantile=0.7,
            clip_score=100,
        ),
        launcher_config=dict(
            num_exps_per_instance=1,
            region='us-west-2',
        ),

        path_loader_class=DictToMDPPathLoader,
        path_loader_kwargs=dict(
            delete_after_loading=True,
            recompute_reward=False,
            obs_key="image_observation",
            demo_paths=demo_paths,
        ),
        add_env_demos=False,
        add_env_offpolicy_data=False,

        load_demos=True,
        load_env_dataset_demos=False,

        normalize_env=False,
        env_class=SawyerRigAffordancesV0,
        env_kwargs=dict(
            env_type="top_drawer",
        ),
        renderer_kwargs=dict(
            create_image_format='HWC',
            output_image_format='CWH',
            flatten_image=True,
            width=48,
            height=48,
        ),
        policy_class = GaussianCNNPolicy,
        qf_class = ConcatCNN,
        vf_class = CNN,
        policy_kwargs = dict(
            # CNN params
            input_width=48,
            input_height=48,
            input_channels=3,
            kernel_sizes=[3, 3, 3],
            n_channels=[16, 16, 16],
            strides=[1, 1, 1],
            hidden_sizes=[1024, 512, 256],
            paddings=[1, 1, 1],
            pool_type='max2d',
            pool_sizes=[2, 2, 1],  # the one at the end means no pool
            pool_strides=[2, 2, 1],
            pool_paddings=[0, 0, 0],
            # Gaussian params
            max_log_std=0,
            min_log_std=-6,
            std_architecture="values",
        ),
        qf_kwargs = dict(
            input_width=48,
            input_height=48,
            input_channels=3,
            kernel_sizes=[3, 3, 3],
            n_channels=[16, 16, 16],
            strides=[1, 1, 1],
            hidden_sizes=[1024, 512, 256],
            paddings=[1, 1, 1],
            pool_type='max2d',
            pool_sizes=[2, 2, 1],  # the one at the end means no pool
            pool_strides=[2, 2, 1],
            pool_paddings=[0, 0, 0],
        ),
        vf_kwargs = dict(
            input_width=48,
            input_height=48,
            input_channels=3,
            kernel_sizes=[3, 3, 3],
            n_channels=[16, 16, 16],
            strides=[1, 1, 1],
            hidden_sizes=[1024, 512, 256],
            paddings=[1, 1, 1],
            pool_type='max2d',
            pool_sizes=[2, 2, 1],  # the one at the end means no pool
            pool_strides=[2, 2, 1],
            pool_paddings=[0, 0, 0],
        ),

        seed=random.randint(0, 100000),
    )

    search_space = {
        "seed": range(5),
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    variants = []
    for variant in sweeper.iterate_hyperparameters():
        variants.append(variant)

    run_variants(experiment, variants, run_id=0, process_args_fn=process_args) #HERE
