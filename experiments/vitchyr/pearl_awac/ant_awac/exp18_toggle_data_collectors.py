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

    print(exp_name)
    exp_id = 0

    def run_sweep(search_space, variant, xid):
        sweeper = hyp.DeterministicHyperparameterSweeper(
            search_space, default_parameters=variant,
        )
        for _, variant in enumerate(sweeper.iterate_hyperparameters()):
            for _ in range(nseeds):
                variant['exp_id'] = xid
                xid += 1
                run_experiment(
                    name_to_exp[variant['tags']['method']],
                    unpack_variant=True,
                    exp_name=exp_name,
                    mode=mode,
                    variant=variant,
                    time_in_mins=3 * 24 * 60 - 1,
                    use_gpu=gpu,
                )
        return xid

    def sac_sweep(xid):
        configs = [
            base_dir / 'configs/default_sac.conf',
            base_dir / 'configs/ant_four_dir.conf',
        ]
        if debug:
            configs.append(base_dir / 'configs/debug.conf')
        variant = ppp.recursive_to_dict(load_pyhocon_configs(configs))

        search_space = {
            'trainer_kwargs.train_context_decoder': [
                True,
                False,
            ],
            'use_data_collectors': [
                True,
                False,
            ],
        }
        return run_sweep(search_space, variant, xid)

    exp_id = sac_sweep(exp_id)
    print(exp_name, exp_id)




if __name__ == "__main__":
    main()

