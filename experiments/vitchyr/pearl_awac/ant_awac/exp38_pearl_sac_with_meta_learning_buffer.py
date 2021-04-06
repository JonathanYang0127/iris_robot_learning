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
            'algo_kwargs.save_replay_buffer': [
                True,
            ],
            'algo_kwargs.save_algorithm': [
                True,
            ],
            'algo_kwargs.save_extra_every_epoch': [
                False,
            ],
            'algo_kwargs.save_extra_manual_epoch_list': [
                [0, 1, 49, 100, 200, 300, 400, 500],
            ],
            'algo_kwargs.num_iterations': [
                500,
            ],
            '_debug_do_not_sqrt': [
                False,
            ],
            'save_video': [
                False,
            ],
            'save_video_period': [
                25,
            ],
            'algo_kwargs.num_iterations_with_reward_supervision': [
                # 10,
                # 20,
                # 30,
                None,
            ],
            'algo_kwargs.use_meta_learning_buffer': [
                True,
            ],
        }.items():
            search_space[k] = v
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
            # base_dir / 'configs/offline_pretraining.conf',
            base_dir / 'configs/ant_four_dir.conf',
            ]
        if debug:
            configs.append(base_dir / 'configs/debug.conf')
        variant = ppp.recursive_to_dict(load_pyhocon_configs(configs))
        search_space = {}
        return run_sweep(search_space, variant, xid)

    exp_id = sac_sweep(exp_id)
    print(exp_name, exp_id)




if __name__ == "__main__":
    main()

