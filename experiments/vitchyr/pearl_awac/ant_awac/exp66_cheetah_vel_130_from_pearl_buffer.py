"""
PEARL Experiment
"""

import click
from pathlib import Path

from rlkit.launchers.launcher_util import run_experiment, load_pyhocon_configs
import rlkit.pythonplusplus as ppp
from rlkit.torch.pearl.cql_launcher import pearl_cql_experiment
import rlkit.misc.hyperparameter as hyp
from rlkit.torch.pearl.sac_launcher import pearl_sac_experiment
from rlkit.torch.pearl.awac_launcher import pearl_awac_experiment


name_to_exp = {
    'CQL': pearl_cql_experiment,
    'AWAC': pearl_awac_experiment,
    'SAC': pearl_sac_experiment,
}


@click.command()
@click.option('--debug', is_flag=True, default=False)
@click.option('--dry', is_flag=True, default=False)
@click.option('--suffix', default=None)
@click.option('--nseeds', default=1)
@click.option('--mode', default='azure')
@click.option('--olddd', is_flag=True, default=False)
def main(debug, dry, suffix, nseeds, mode, olddd):
    gpu = True

    base_dir = Path(__file__).parent.parent

    path_parts = __file__.split('/')
    suffix = '' if suffix is None else '--{}'.format(suffix)
    exp_name = 'pearl-awac-{}--{}{}'.format(
        path_parts[-2].replace('_', '-'),
        path_parts[-1].split('.')[0].replace('_', '-'),
        suffix,
    )

    if debug or dry:
        exp_name = 'dev--' + exp_name
        mode = 'local'
        nseeds = 1

    print(exp_name)

    if mode == 'local':
        remote_mount_configs = [
            dict(
                local_dir='/home/vitchyr/mnt3/azure/',
                mount_point='/preloaded_data',
            ),
            # dict(
            #     local_dir='/home/vitchyr/mnt2/log2/',
            #     mount_point='/preloaded_data2',
            # ),
        ]
    elif mode == 'azure':
        remote_mount_configs = [
            dict(
                local_dir='/doodad_tmp/',
                mount_point='/preloaded_data',
            ),
        ]
    elif mode == 'here_no_doodad':
        remote_mount_configs = []
    else:
        raise ValueError(mode)

    def run_sweep(variant):
        search_space = {
            'load_buffer_kwargs.pretrain_buffer_path': [
                "/preloaded_data/21-05-05_pearl-awac-ant-awac--exp59-half-cheetah-130-online-pearl/16h-02m-49s_run2/extra_snapshot_itr50.cpkl",
            ],
            'saved_tasks_path': [
                "/preloaded_data/21-05-05_pearl-awac-ant-awac--exp59-half-cheetah-130-online-pearl/16h-02m-49s_run2/tasks_description.joblib",
            ],
            # 'load_buffer_kwargs.start_idx': [
            #     -100000,
            # ],
            # 'load_buffer_kwargs.end_idx': [
            #     200000
            # ],
            'trainer_kwargs.train_context_decoder': [
                True,
            ],
            'trainer_kwargs.backprop_q_loss_into_encoder': [
                False,
                True,
            ],
            'algo_kwargs.num_iterations_with_reward_supervision': [
                0,
            ],
            'algo_kwargs.exploration_resample_latent_period': [
                1,
            ],
            'online_trainer_kwargs.awr_weight': [
                1.0,
            ],
            'online_trainer_kwargs.reparam_weight': [
                1.0,
            ],
            'online_trainer_kwargs.use_reparam_update': [
                True,
            ],
            'online_trainer_kwargs.use_awr_update': [
                True,
            ],
            'tags.encoder_buffer_mode': [
                'match_rl',
            ],
            'seed': list(range(nseeds)),
            'trainer_kwargs.beta': [
                100,
            ],
        }
        if not olddd:
            from rlkit.launchers.doodad_wrapper import run_experiment
            run_experiment(
                name_to_exp[variant['tags']['method']],
                params=search_space,
                default_params=variant,
                exp_name=exp_name,
                mode=mode,
                use_gpu=gpu,
                non_code_dirs_to_mount=[
                    dict(
                        local_dir='/home/vitchyr/.mujoco/',
                        mount_point='/root/.mujoco',
                    ),
                ],
                remote_mount_configs=remote_mount_configs,
                start_run_id=2,
            )
        else:
            from rlkit.launchers.launcher_util import run_experiment
            sweeper = hyp.DeterministicHyperparameterSweeper(
                search_space, default_parameters=variant,
            )
            for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
                variant['expd_id'] = exp_id
                run_experiment(
                    name_to_exp[variant['tags']['method']],
                    unpack_variant=True,
                    exp_name=exp_name,
                    mode=mode,
                    variant=variant,
                    time_in_mins=3 * 24 * 60 - 1,
                    use_gpu=gpu,
                )

    configs = [
        base_dir / 'configs/default_awac.conf',
        base_dir / 'configs/offline_pretraining.conf',
        base_dir / 'configs/half_cheetah_130_offline.conf',
        ]
    if debug:
        configs.append(base_dir / 'configs/debug.conf')
    variant = ppp.recursive_to_dict(load_pyhocon_configs(configs))
    run_sweep(variant)

    print(exp_name)




if __name__ == "__main__":
    main()

