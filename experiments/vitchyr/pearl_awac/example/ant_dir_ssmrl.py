"""
PEARL Experiment
"""

from pathlib import Path

import click

import rlkit.pythonplusplus as ppp
from rlkit.launchers.launcher_util import load_pyhocon_configs
from rlkit.torch.pearl.awac_launcher import pearl_awac_experiment
from rlkit.launchers.doodad_wrapper import run_experiment


@click.command()
@click.option('--debug', is_flag=True, default=False)
@click.option('--dry', is_flag=True, default=False)
@click.option('--suffix', default=None)
@click.option('--nseeds', default=1)
def main(debug, dry, suffix, nseeds):
    mode = 'sss'
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

    configs = [
        base_dir / 'configs/default_awac.conf',
        base_dir / 'configs/offline_pretraining.conf',
        base_dir / 'configs/ant_four_dir_offline.conf',
    ]
    if debug:
        configs.append(base_dir / 'configs/debug.conf')
    variant = ppp.recursive_to_dict(load_pyhocon_configs(configs))

    PATH_TO_MACAW_BUFFERS = None  # TODO: fill in

    search_space = {
        'seed': list(range(nseeds)),
        'trainer_kwargs.beta': [
            100,
        ],
        'macaw_format_base_path': [
            PATH_TO_MACAW_BUFFERS if mode == 'here_no_doodad' else '/macaw_buffer'
        ],
        'load_buffer_kwargs.is_macaw_buffer_path': [
            True
        ],
        'trainer_kwargs.train_context_decoder': [
            True,
        ],
        'trainer_kwargs.backprop_q_loss_into_encoder': [
            False,
        ],
        'train_task_idxs': [
            [0, 1, 2, 3],
        ],
        'eval_task_idxs': [
            [5, 6, 7, 8],
        ],
        'algo_kwargs.num_iterations_with_reward_supervision': [
            0,
        ],
        'algo_kwargs.exploration_resample_latent_period': [
            1,
        ],
        'algo_kwargs.encoder_buffer_matches_rl_buffer': [
            True,
        ],
        'algo_kwargs.freeze_encoder_buffer_in_unsupervised_phase': [
            False,
        ],
        'algo_kwargs.clear_encoder_buffer_before_every_update': [
            False,
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
    }

    print(exp_name)
    run_experiment(
        pearl_awac_experiment,
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
            dict(
                local_dir=PATH_TO_MACAW_BUFFERS,
                mount_point='/macaw_data',
            ),
        ],
    )
    print(exp_name)


if __name__ == "__main__":
    main()
