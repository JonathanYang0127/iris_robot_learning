import click
import joblib
from pathlib import Path
import pickle
import sys

from rlkit.misc.asset_loader import (
    load_local_or_remote_file,
)
import numpy as np
import json
from rlkit.data_management.offline_dataset.util import (
    rlkit_buffer_to_macaw_format,
    rlkit_buffer_to_borel_format,
)


def get_task_idx(tasks, variant):
    fixed_task = variant['env_params']['fixed_tasks']
    for i, task in enumerate(tasks):
        if task['goal'] == fixed_task[0]:
            return i
    import ipdb; ipdb.set_trace()
    pass


def many_buffers_to_macaw_format(
        exps_dir,
        tasks_path,
        save_dir,
        snapshot_iteration,
        path_length=200,
        discount_factor=0.99,
        output_format='macaw',
):
    """
    Given a directory of the format

    exps_dir/
        exp0/
            extra_snapshot_itrX.cpkl
            variant.json
        exp1/
            extra_snapshot_itrX.cpkl
            variant.json
        ...

    and a path to `tasks.pkl`
    this generated a directory of the format

    save_dir/
        tasks.pkl
        macaw_buffer_task_0.npy
        macaw_buffer_task_1.npy
        ...

    :param exps_dir:
    :param tasks_path:
    :param save_dir:
    :param snapshot_iteration:
    :return:
    """
    exps_path = Path(exps_dir)
    assert output_format in {'macaw', 'borel'}
    save_dir = Path(save_dir) / '{}_buffer'.format(output_format)
    save_dir.mkdir(exist_ok=True)
    tasks = pickle.load(open(tasks_path, 'rb'))
    pickle.dump(tasks, open(save_dir / 'tasks.pkl', 'wb'))

    metadata_save_path = save_dir / 'data_generation_info.txt'
    with open(metadata_save_path, 'w') as f:
        f.write("script: {}\n".format(' '.join(sys.argv)))
        f.write("path_length = {}\n".format(path_length))
        f.write("discount_factor = {}\n".format(discount_factor))
        f.write("snapshot_iteration = {}\n".format(snapshot_iteration))
        f.write("exps_dir = '{}'\n".format(exps_dir))
        f.write("tasks_path = '{}'\n".format(tasks_path))
        f.write("save_dir = '{}'\n".format(save_dir))
    print("saved metadata to {}".format(metadata_save_path))

    task_idx_to_snapshot_path = {}
    for subdir in exps_path.iterdir():
        snapshot_path = subdir / 'extra_snapshot_itr{}.cpkl'.format(snapshot_iteration)
        variant_path = subdir / 'variant.json'
        variant = json.load(open(variant_path, 'r'))

        task_idx = get_task_idx(tasks, variant)
        task_idx_to_snapshot_path[task_idx] = snapshot_path

    for task_idx in range(len(tasks)):
        if task_idx not in task_idx_to_snapshot_path:
            continue
        snapshot_path = task_idx_to_snapshot_path[task_idx]
        snapshot = joblib.load(snapshot_path)
        saved_replay_buffer = snapshot['replay_buffer']
        buffer = saved_replay_buffer.task_buffers[0]
        if output_format == 'macaw':
            buffer = rlkit_buffer_to_macaw_format(buffer, discount_factor, path_length=path_length)
        else:
            buffer = rlkit_buffer_to_borel_format(buffer, discount_factor, path_length=path_length)
        save_path = str(
            save_dir / '{}_buffer_task_{}.npy'.format(output_format, task_idx)
        )
        print('saving to', save_path)
        np.save(save_path, buffer)


@click.command()
@click.option(
    '--exps_dir',
    default='/home/vitchyr/mnt2/log2/21-04-27-pearl-awac-ant-awac--exp47-train-ant-indiv-many-directions-brc-every-10/',
)
@click.option(
    '--tasks_path',
    default='/home/vitchyr/mnt2/log2/demos/ant_dir/tasks/ant_32_tasks.pkl',
)
@click.option(
    '--save_dir',
    default='/home/vitchyr/mnt2/log2/demos/ant_dir_32/',
)
@click.option(
    '--format',
    default='macaw',
)
@click.option(
    '--iteration',
    default=20,
)
def main(exps_dir, tasks_path, save_dir, format, iteration):
    many_buffers_to_macaw_format(
        exps_dir, tasks_path, save_dir,
        iteration,
        output_format=format,
        path_length=200,
        discount_factor=0.99,
    )


if __name__ == '__main__':
    main()