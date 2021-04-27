import joblib
from pathlib import Path
import pickle

from rlkit.misc.asset_loader import (
    load_local_or_remote_file,
)
import numpy as np
import json
from rlkit.data_management.offline_dataset.util import (
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
        exps_path,
        tasks_path,
        save_dir,
        snapshot_iteration,
):
    """
    Given a directory of the format

    exps_path/
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

    :param exps_path:
    :param tasks_path:
    :param save_dir:
    :param snapshot_iteration:
    :return:
    """
    exps_path = Path(exps_path)
    save_dir = Path(save_dir) / 'borel_buffer'
    save_dir.mkdir(exist_ok=True)
    tasks = pickle.load(open(tasks_path, 'rb'))
    pickle.dump(tasks, open(save_dir / 'tasks.pkl', 'wb'))

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
        macaw_buffer = rlkit_buffer_to_borel_format(buffer, discount_factor, path_length=path_length)
        save_path = str(
            save_dir / 'borel_buffer_task_{}.npy'.format(task_idx)
        )
        print('saving to', save_path)
        np.save(save_path, macaw_buffer)


if __name__ == '__main__':
    path_length = 200
    discount_factor = 0.99
    pretrain_buffer_path = "21-02-22-ant-awac--exp7-ant-dir-4-eval-4-train-sac-to-get-buffer-longer/21-02-22-ant-awac--exp7-ant-dir-4-eval-4-train-sac-to-get-buffer-longer_2021_02_23_06_09_23_id000--s270987/extra_snapshot_itr400.cpkl"
    exps_dir = '/home/vitchyr/mnt2/log2/21-04-26-pearl-awac-ant-awac--exp45-train-ant-indiv-many-directions-brc--take2/'
    tasks_path = '/home/vitchyr/mnt2/log2/demos/ant_dir/tasks/ant_32_tasks.pkl'
    save_dir = '/home/vitchyr/mnt2/log2/demos/ant_dir_32/'
    snapshot_iteration = 0
    many_buffers_to_macaw_format(exps_dir, tasks_path, save_dir, snapshot_iteration)
