"""
PEARL Experiment
"""

import click
from pathlib import Path

from rlkit.launchers.launcher_util import load_pyhocon_configs
from rlkit.launchers.doodad_wrapper import run_experiment
import rlkit.pythonplusplus as ppp
from rlkit.torch.pearl.cql_launcher import pearl_cql_experiment
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
        # nseeds = 1

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
            'trainer_kwargs.train_context_decoder': [
                True,
            ],
            'use_data_collectors': [
                False,
            ],
            'n_eval_tasks': [
                8,
            ],
            'n_train_tasks': [
                4,
            ],
            'env_params.n_tasks': [
                8,
            ],
            'env_params.direction_in_degrees': [
                True,
            ],
            'env_params.fixed_tasks': [
                # [
                #     22.5,
                #     112.5,
                #     202.5,
                #     292.5,
                #     0,
                #     90,
                #     180,
                #     270,
                # ],
                [
                    67.5,
                    157.5,
                    247.5,
                    337.5,
                    0,
                    90,
                    180,
                    270,
                ],
                # [
                #     45,
                #     135,
                #     225,
                #     315,
                #     0,
                #     90,
                #     180,
                #     270,
                # ],
                # [
                #     0,
                #     90,
                #     180,
                #     270,
                #     45,
                #     135,
                #     225,
                #     315,
                # ],
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
            start_run_id=8,
        )

    sac_sweep()



if __name__ == "__main__":
    main()

