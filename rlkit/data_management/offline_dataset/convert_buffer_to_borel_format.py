from pathlib import Path

from rlkit.data_management.offline_dataset.util import (
    rlkit_buffer_to_borel_format
)
from rlkit.misc.asset_loader import (
    load_local_or_remote_file,
    local_path_from_s3_or_local_path,
)
import numpy as np


if __name__ == '__main__':
    path_length = 200
    discount_factor = 0.99
    pretrain_buffer_path = "21-02-22-ant-awac--exp7-ant-dir-4-eval-4-train-sac-to-get-buffer-longer/21-02-22-ant-awac--exp7-ant-dir-4-eval-4-train-sac-to-get-buffer-longer_2021_02_23_06_09_23_id000--s270987/extra_snapshot_itr400.cpkl"
    data = load_local_or_remote_file(
        pretrain_buffer_path,
        file_type='joblib',
    )
    saved_replay_buffer = data['replay_buffer']
    save_dir = Path(
        local_path_from_s3_or_local_path(pretrain_buffer_path)
    ).parent / 'borel_buffer'
    save_dir.mkdir(exist_ok=True)
    for k in saved_replay_buffer.task_buffers:
        buffer = saved_replay_buffer.task_buffers[k]
        data = rlkit_buffer_to_borel_format(buffer, discount_factor, path_length)
        save_path = save_dir / 'borel_buffer_task_{}.npy'.format(k)
        print('saving to', save_path)
        np.save(save_path, data)
