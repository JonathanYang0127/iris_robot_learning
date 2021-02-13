from collections import OrderedDict, defaultdict

from rlkit.core.logging import append_log
from rlkit.envs.pearl_envs import HalfCheetahDirEnv
from rlkit.misc import eval_util
from rlkit.misc.eval_util import create_stats_ordered_dict


def make_named_path_compatible(fn, divider='/'):
    """
    converts a function of type

    def f(paths) -> dictionary

    to

    def f(Dict[str, List]) -> dictionary

    with the dictionary key prefixed

    for example

    ```
    def foo(paths):
        return {
            'num_paths': len(paths)
        }

    paths = [1,2]

    print(foo(paths))
    # prints {'num_paths': 2}

    named_paths = {
        'a': [1,2],
        'b': [1,2],
    }
    new_foo = make_named_path_compatible(foo)

    print(new_foo(paths))
    # prints {'a/num_paths': 2, 'b'/num_paths': 1}
    ```
    """

    def unpacked_fn(named_paths):
        results = OrderedDict()
        for name, paths in named_paths.items():
            new_results = fn(paths)
            append_log(results, new_results, prefix=name, divider=divider)
        return results

    return unpacked_fn


def get_diagnostics(env):
    diagnostics = [
        eval_util.get_generic_path_information,
    ]
    if isinstance(env, HalfCheetahDirEnv):
        diagnostics.append(half_cheetah_dir_diagnostics)
    return [
        make_named_path_compatible(fn) for fn in
        diagnostics
    ]


def half_cheetah_dir_diagnostics(paths):
    statistics = OrderedDict()
    stat_to_lists = defaultdict(list)
    for path in paths:
        for k in ['reward_forward', 'reward_ctrl']:
            stats_for_this_path = []
            for env_info in path['env_infos']:
                stats_for_this_path.append(env_info[k])
            stat_to_lists[k].append(stats_for_this_path)
    for stat_name, stat_list in stat_to_lists.items():
        statistics.update(create_stats_ordered_dict(
            stat_name,
            stat_list,
            always_show_all_stats=True,
        ))
        statistics.update(create_stats_ordered_dict(
            '{}/final'.format(stat_name),
            [s[-1:] for s in stat_list],
            always_show_all_stats=True,
            exclude_max_min=True,
        ))
    return statistics
