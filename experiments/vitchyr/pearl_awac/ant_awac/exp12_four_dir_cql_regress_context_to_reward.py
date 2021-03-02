"""
PEARL Experiment
"""

import click
from pathlib import Path

from rlkit.launchers.launcher_util import run_experiment, load_pyhocon_configs
import rlkit.pythonplusplus as ppp
from rlkit.torch.pearl.cql_launcher import pearl_cql_experiment
import rlkit.misc.hyperparameter as hyp


@click.command()
@click.option('--debug', is_flag=True, default=False)
@click.option('--dry', is_flag=True, default=False)
@click.option('--suffix', default=None)
@click.option('--nseeds', default=3)
def main(debug, dry, suffix, nseeds):
    mode = 'sss'
    gpu = True

    base_dir = Path(__file__).parent.parent
    configs = [
        base_dir / 'configs/default_cql.conf',
        base_dir / 'configs/offline_pretraining.conf',
        base_dir / 'configs/short_fine_tuning.conf',
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
        configs.append(base_dir / 'configs/debug.conf')
    if debug or dry:
        exp_name = 'dev--' + exp_name
        mode = 'local'
        nseeds = 1

    config = load_pyhocon_configs(configs)
    variant = ppp.recursive_to_dict(config)

    search_space = {
        'trainer_kwargs.train_context_decoder': [
            True,
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(nseeds):
            variant['exp_id'] = exp_id
            run_experiment(
                pearl_cql_experiment,
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

