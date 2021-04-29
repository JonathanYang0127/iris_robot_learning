Step 0.
(only for ant task)
Run `make_tasks.py` to generate a `tasks.pkl` file.

Step 1.
Run `ant_dir_data_generation.py`. This will save a directory of the format

```
exp_dir/
    run0/
        extra_snapshot_itr0.cpkl
        extra_snapshot_itr10.cpkl
    run1/
        extra_snapshot_itr0.cpkl
        extra_snapshot_itr10.cpkl
    run2/
        ...
```

Step 2.
Convert this data into a replay buffer
```

python rlkit/data_management/offline_dataset/convert_multiple_rlkit_buffers_to_standard_format.py \
    --exps_dir=path/to/exps_dir
    --task_path=path/to/tasks.pkl
    --save_dir=path/to/logs/
    --iteration=20  # or whatever iteration you want
```
where `path/to/tasks.pkl` is a `pickle`'ed list of tasks. You need to get this `pickle` file from elsewhere (e.g. see step 0).

Step 3.
Inside `ant_dir_ssmrl.py`, modify `PATH_TO_MACAW_BUFFERS` to point to `path/to/logs/macaw_buffer`

Step 4.
Run `ant_dir_ssmrl.py`
