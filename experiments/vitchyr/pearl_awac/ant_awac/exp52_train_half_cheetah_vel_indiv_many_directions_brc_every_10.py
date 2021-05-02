"""
PEARL Experiment
"""

import click
from pathlib import Path

from rlkit.launchers.launcher_util import load_pyhocon_configs
import rlkit.pythonplusplus as ppp
from rlkit.torch.pearl.sac_launcher import pearl_sac_experiment
import rlkit.misc.hyperparameter as hyp
import pickle



@click.command()
@click.option('--debug', is_flag=True, default=False)
@click.option('--dry', is_flag=True, default=False)
@click.option('--suffix', default=None)
@click.option('--nseeds', default=1)
@click.option('--mode', default='local')
@click.option('--olddd', is_flag=True, default=False)
def main(debug, dry, suffix, nseeds, mode, olddd):
    gpu = False
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
            base_dir / 'configs/half_cheetah_one_velocity.conf',
        ]
        if debug:
            configs.append(base_dir / 'configs/debug.conf')
        variant = ppp.recursive_to_dict(load_pyhocon_configs(configs))

        tasks = pickle.load(open('/home/vitchyr/mnt2/log2/demos/half_cheetah_vel_130/half_cheetah_vel_130_tasks.pkl', 'rb'))

        search_space = {
            'use_data_collectors': [
                False,
            ],
            'n_eval_tasks': [
                1,
            ],
            'n_train_tasks': [
                1,
            ],
            'env_params.presampled_tasks': [
                [task] for task in tasks
            ],
            'seed': list(range(nseeds)),
        }
        if olddd:
            from rlkit.launchers.launcher_util import run_experiment
            sweeper = hyp.DeterministicHyperparameterSweeper(
                search_space, default_parameters=variant,
            )
            for _, variant in enumerate(sweeper.iterate_hyperparameters()):
                for _ in range(nseeds):
                    run_experiment(
                        method_call=pearl_sac_experiment,
                        unpack_variant=True,
                        exp_name=exp_name,
                        mode=mode,
                        variant=variant,
                        time_in_mins=3 * 24 * 60 - 1,
                        use_gpu=gpu,
                    )
        else:
            from rlkit.launchers.doodad_wrapper import run_experiment
            run_experiment(
                method_call=pearl_sac_experiment,
                params=search_space,
                default_params=variant,
                exp_name=exp_name,
                mode=mode,
                use_gpu=gpu,
                start_run_id=0,
                azure_region='westus',
            )

    sac_sweep()
    print(exp_name)



if __name__ == "__main__":
    main()

