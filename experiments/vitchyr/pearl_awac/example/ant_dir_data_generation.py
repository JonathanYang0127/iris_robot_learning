"""
SAC Experiment
"""

import click
from pathlib import Path

from rlkit.launchers.launcher_util import load_pyhocon_configs
from rlkit.launchers.doodad_wrapper import run_experiment
import rlkit.pythonplusplus as ppp
from rlkit.torch.pearl.sac_launcher import pearl_sac_experiment


@click.command()
@click.option('--debug', is_flag=True, default=False)
@click.option('--dry', is_flag=True, default=False)
@click.option('--suffix', default=None)
@click.option('--nseeds', default=1)
@click.option('--mode', default='local')
def main(debug, dry, suffix, nseeds, mode):
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

    if dry:
        nseeds = 1
        mode = 'here_no_doodad'

    print(exp_name)

    def sac_sweep():
        configs = [
            base_dir / 'configs/default_sac.conf',
            base_dir / 'configs/ant.conf',
        ]
        if debug:
            configs.append(base_dir / 'configs/debug.conf')
        variant = ppp.recursive_to_dict(load_pyhocon_configs(configs))

        search_space = {
            'algo_kwargs.meta_batch': [2],
            'algo_kwargs.embedding_batch_size': [2],
            'algo_kwargs.embedding_mini_batch_size': [2],
            'algo_kwargs.num_iterations': [
                51,
            ],
            'algo_kwargs.save_extra_manual_epoch_list': [
                [0, 10, 20, 30, 40, 50],
            ],
            'n_eval_tasks': [
                1,
            ],
            'n_train_tasks': [
                1,
            ],
            'env_params.n_tasks': [
                1,
            ],
            'env_params.direction_in_degrees': [
                True,
            ],
            'env_params.fixed_tasks': [
                [0],
                [90],
                [180],
                [270],
                [45],
                [135],
                [225],
                [315],
                [22.5],
                [112.5],
                [202.5],
                [292.5],
                [67.5],
                [157.5],
                [247.5],
                [337.5],
                [11.25],
                [101.25],
                [191.25],
                [281.25],
                [56.25],
                [146.25],
                [236.25],
                [326.25],
                [33.75],
                [123.75],
                [213.75],
                [303.75],
                [78.75],
                [168.75],
                [258.75],
                [348.75],
            ],
            'trainer_kwargs.train_context_decoder': [
                # False,
                True,
            ],
            'trainer_kwargs.backprop_q_loss_into_encoder': [
                True,
                # False,
            ],
            'use_data_collectors': [
                False,
            ],
            'seed': list(range(nseeds)),
        }
        run_experiment(
            method_call=pearl_sac_experiment,
            params=search_space,
            default_params=variant,
            exp_name=exp_name,
            mode=mode,
            use_gpu=gpu,
            start_run_id=0,
        )

    sac_sweep()



if __name__ == "__main__":
    main()

