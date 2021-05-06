from pathlib import Path

import numpy as np

from rlkit.data_management.offline_dataset.util import (
    rlkit_buffer_to_macaw_format,
    rlkit_buffer_to_borel_format,
)
from rlkit.misc.asset_loader import (
    load_local_or_remote_file,
)

if __name__ == '__main__':
    path_length = 200
    discount_factor = 0.99
    output_format = 'macaw'
    pretrain_buffer_path = "/home/vitchyr/mnt2/log2/21-02-22-ant-awac--exp7-ant-dir-4-eval-4-train-sac-to-get-buffer-longer/21-02-22-ant-awac--exp7-ant-dir-4-eval-4-train-sac-to-get-buffer-longer_2021_02_23_06_09_23_id000--s270987/extra_snapshot_itr400.cpkl"
    save_dir = Path(pretrain_buffer_path).parent
    save_dir = Path(save_dir) / '{}_buffer'.format(output_format)
    save_dir.mkdir(parents=True, exist_ok=True)
    snapshot = load_local_or_remote_file(
        pretrain_buffer_path,
        file_type='joblib',
    )
    # saved_replay_buffer = data['replay_buffer']
    # save_dir = Path(
    #     local_path_from_s3_or_local_path(pretrain_buffer_path)
    # ).parent / 'macaw_buffer'
    # save_dir.mkdir(exist_ok=True)
    # for k in saved_replay_buffer.task_buffers:
    #     buffer = saved_replay_buffer.task_buffers[k]
    #     data = rlkit_buffer_to_macaw_format(buffer, discount_factor, path_length)
    #     save_path = save_dir / 'macaw_buffer_task_{}.npy'.format(k)
    #     print('saving to', save_path)
    #     np.save(save_path, data)

    for key in ['replay_buffer', 'enc_replay_buffer']:
        saved_replay_buffer = snapshot[key]
        for task_idx in saved_replay_buffer.task_buffers:
            buffer = saved_replay_buffer.task_buffers[task_idx]
            if output_format == 'macaw':
                buffer = rlkit_buffer_to_macaw_format(
                    buffer, discount_factor, path_length=path_length,
                )
            else:
                buffer = rlkit_buffer_to_borel_format(
                    buffer, discount_factor, path_length=path_length,
                )
            save_path = str(
                save_dir / '{}_{}_task_{}.npy'.format(output_format, key, task_idx)
            )
            print('saving to', save_path)
            np.save(save_path, buffer)
