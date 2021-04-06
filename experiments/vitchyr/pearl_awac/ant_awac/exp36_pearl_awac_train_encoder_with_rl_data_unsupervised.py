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
        for k, v in {
            'load_buffer_kwargs.start_idx': [
                -100000,
            ],
            'load_buffer_kwargs.end_idx': [
                200000
            ],
            'trainer_kwargs.train_context_decoder': [
                True,
            ],
            'trainer_kwargs.backprop_q_loss_into_encoder': [
                False,
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
        }.items():
            search_space[k] = v
        sweeper = hyp.DeterministicHyperparameterSweeper(
            search_space, default_parameters=variant,
        )
        for _, variant in enumerate(sweeper.iterate_hyperparameters()):
            for _ in range(nseeds):
                if (
                        not variant['online_trainer_kwargs']['use_awr_update']
                        and not variant['online_trainer_kwargs']['use_reparam_update']
                ):
                    continue
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

    def cql_sweep(xid):
        configs = [
            base_dir / 'configs/default_cql.conf',
            base_dir / 'configs/offline_pretraining.conf',
            base_dir / 'configs/ant_four_dir_offline.conf',
            ]
        if debug:
            configs.append(base_dir / 'configs/debug.conf')
        variant = ppp.recursive_to_dict(load_pyhocon_configs(configs))
        search_space = {
            'trainer_kwargs.with_lagrange': [
                True,
            ],
            'trainer_kwargs.min_q_weight': [
                10.0,
            ],
        }
        return run_sweep(search_space, variant, xid)

    def awac_sweep(xid):
        configs = [
            base_dir / 'configs/default_awac.conf',
            base_dir / 'configs/offline_pretraining.conf',
            base_dir / 'configs/ant_four_dir_offline.conf',
            ]
        if debug:
            configs.append(base_dir / 'configs/debug.conf')
        variant = ppp.recursive_to_dict(load_pyhocon_configs(configs))
        search_space = {
            'trainer_kwargs.beta': [
                100,
            ],
        }
        return run_sweep(search_space, variant, xid)

    # exp_id = cql_sweep(exp_id)
    exp_id = awac_sweep(exp_id)
    print(exp_name, exp_id)




if __name__ == "__main__":
    main()

