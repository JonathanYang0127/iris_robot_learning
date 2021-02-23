import click
from pathlib import Path

from rlkit.launchers.launcher_util import run_experiment
import rlkit.pythonplusplus as ppp
from rlkit.torch.pearl.new_sac_launcher import pearl_sac_experiment
import rlkit.misc.hyperparameter as hyp


def load_configs(config_paths):
    from pyhocon import ConfigFactory, ConfigTree
    config = ConfigFactory.parse_file(config_paths[0])
    for path in config_paths[1:]:
        new_config = ConfigFactory.parse_file(path)
        config = ConfigTree.merge_configs(config, new_config)
    return config


@click.command()
@click.option('--debug', is_flag=True, default=False)
@click.option('--dry', is_flag=True, default=False)
@click.option('--suffix', default=None)
def main(debug, dry, suffix):
    mode = 'sss'
    n_seeds = 3
    gpu = True

    base_dir = Path(__file__).parent
    configs = [
        base_dir / 'configs/default_sac.conf',
        base_dir / 'configs/ant_four_dir.conf',
    ]

    path_parts = __file__.split('/')
    suffix = '' if suffix is None else '--{}'.format(suffix)
    exp_name = '{}--{}{}'.format(
        path_parts[-2].replace('_', '-'),
        path_parts[-1].split('.')[0].replace('_', '-'),
        suffix,
    )
    if debug:
        configs.append(base_dir / 'configs/debug.conf')
    if debug or dry:
        exp_name = 'dev--' + exp_name
        mode = 'local'
        n_seeds = 1

    config = load_configs(configs)
    variant = ppp.recursive_to_dict(config)

    search_space = {
        'n_train_tasks_for_video': [4],
        'n_test_tasks_for_video': [4],
        'save_video_kwargs.video_img_size': [64],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            variant['exp_id'] = exp_id
            run_experiment(
                pearl_sac_experiment,
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

