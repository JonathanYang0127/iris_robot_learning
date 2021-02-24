"""
PEARL Experiment
"""

import click
from pathlib import Path

from rlkit.launchers.launcher_util import run_experiment, load_pyhocon_configs
import rlkit.pythonplusplus as ppp
from rlkit.torch.pearl.awac_launcher import pearl_awac_launcher_simple
import rlkit.misc.hyperparameter as hyp


@click.command()
@click.option('--debug', is_flag=True, default=False)
@click.option('--dry', is_flag=True, default=False)
@click.option('--suffix', default=None)
def main(debug, dry, suffix):
    mode = 'sss'
    n_seeds = 3
    gpu = True

    base_dir = Path(__file__).parent.parent
    configs = [
        base_dir / 'configs/default_awac.conf',
        base_dir / 'configs/ant_four_dir_offline.conf',
    ]

    path_parts = __file__.split('/')
    suffix = '' if suffix is None else '--{}'.format(suffix)
    exp_name = 'pearl-awac-{}--{}{}'.format(
        path_parts[-2].replace('_', '-'),
        path_parts[-1].split('.')[0].replace('_', '-'),
        suffix,
    )
    if debug:
        configs.append(base_dir / 'configs/debug_awac.conf')
    if debug or dry:
        exp_name = 'dev--' + exp_name
        mode = 'local'
        n_seeds = 1

    config = load_pyhocon_configs(configs)
    variant = ppp.recursive_to_dict(config)

    search_space = {
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            variant['exp_id'] = exp_id
            run_experiment(
                pearl_awac_launcher_simple,
                unpack_variant=True,
                exp_name=exp_name,
                mode=mode,
                variant=variant,
                time_in_mins=3 * 24 * 60 - 1,
                use_gpu=gpu,
            )
    print(exp_name)


if __name__ == "__main__":
    main()

