"""
Example of running stuff on azure with easy_launch
"""
from doodad.wrappers import easy_launch


def example_function(doodad_config, variant):
    x = variant['x']
    y = variant['y']
    z = variant['z']
    with open(doodad_config.output_directory + '/function_output.txt', "w") as f:
        f.write('sum = {}'.format(x+y+z))
    print('x, y, z = ', x, y, z)
    easy_launch.save_doodad_config(doodad_config)


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
        easy_launch.sweep_function(
            example_function,
            params_to_sweep,
            default_params=default_params,
            log_path='test_gpu_easy_launch_{}'.format(mode),
            mode=mode,
            use_gpu=True,
        )
