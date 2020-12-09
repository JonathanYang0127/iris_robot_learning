"""
PEARL Experiment
"""

import click
import json
import os

import rlkit.misc.hyperparameter as hyp
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.pearl import configs
import rlkit.pythonplusplus as ppp
from rlkit.torch.pearl.launcher import pearl_experiment


@click.command()
@click.argument('config', default=None)
# @click.option('--gpu', default=0)
# @click.option('--docker', is_flag=True, default=False)
# @click.option('--debug', is_flag=True, default=False)
@click.option('--exp_name', default='dev')
def main(config, exp_name):
    with open(os.path.join(config)) as f:
        exp_params = json.load(f)
    variant = ppp.merge_recursive_dicts(
        exp_params,
        configs.default_config,
        ignore_duplicate_keys_in_second_dict=True,
    )

    n_seeds = 1
    mode = 'local'
    # exp_name = 'dev'

    # n_seeds = 5
    # mode = 'sss'
    # exp_name = 'rlkit-half-cheetah-online'

    run_experiment(
        pearl_experiment,
        unpack_variant=False,
        exp_name=exp_name,
        mode=mode,
        variant=variant,
        time_in_mins=int(2.8 * 24 * 60),  # if you use mode=sss
    )
    #
    # search_space = {
    #     'env': [
    #         'half-cheetah',
    #         # 'inv-double-pendulum',
    #         # 'pendulum',
    #         # 'ant',
    #         # 'walker',
    #         # 'hopper',
    #         # 'humanoid',
    #         # 'swimmer',
    #     ],
    # }
    # sweeper = hyp.DeterministicHyperparameterSweeper(
    #     search_space, default_parameters=variant,
    # )
    # for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
    #     for _ in range(n_seeds):
    #         variant['exp_id'] = exp_id

if __name__ == "__main__":
    main()

