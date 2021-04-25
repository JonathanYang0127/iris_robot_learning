"""
Example of running stuff with doodad_wrapper.run_experiment
"""
from rlkit.launchers.doodad_wrapper import run_experiment


def example_function(x, y, z):
    print('x, y, z = ', x, y, z)
    from rlkit.core import logger
    logger.log('hello')
    logger.record_tabular('x', x)
    logger.record_tabular('y', y)
    logger.record_tabular('z', z)
    logger.dump_tabular()


if __name__ == "__main__":
    params_to_sweep = {
        'x': [1, 4],
        'y': [100],
    }
    default_params = {
        'z': 10,
    }
    for mode in [
        # 'here_no_doodad',
        'local',
        # 'azure',
    ]:
        run_experiment(
            example_function,
            params_to_sweep,
            default_params=default_params,
            exp_name='test_gpu_easy_launch_{}'.format(mode),
            mode=mode,
            use_gpu=True,
        )
