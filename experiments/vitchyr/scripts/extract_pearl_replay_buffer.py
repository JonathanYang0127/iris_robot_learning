import argparse
import json
from pathlib import Path

import cloudpickle
import numpy as np
import torch

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.multitask_replay_buffer import MultiTaskReplayBuffer
from rlkit.envs.images import GymEnvRenderer
# from rlkit.envs.images.text_renderer import TextRenderer
from rlkit.envs.images.plot_renderer import TextRenderer, ScrollingPlotRenderer
from rlkit.envs.pearl_envs import HalfCheetahDirEnv, AntDirEnv
from rlkit.envs.wrappers.flat_to_dict import FlatToDictEnv
from rlkit.misc.asset_loader import load_local_or_remote_file
from rlkit.torch.pearl.buffer import PearlReplayBuffer
from rlkit.torch.pearl.diagnostics import (
    DebugInsertImagesEnv,
    FlatToDictPearlPolicy,
)
from rlkit.torch.pearl.launcher_util import load_buffer_onto_algo
from rlkit.torch.pearl.sampler import rollout, rollout_multiple_and_flatten
from rlkit.visualization.video import dump_video
import pickle
import joblib

counter = 0


def extract(args):
    if args.pause:
        import ipdb; ipdb.set_trace()
    # buffer_path = '/home/vitchyr/mnt2/log2/21-02-16-new-pearl--tmp-hc-create-buffer/21-02-16-new-pearl--tmp-hc-create-buffer_2021_02_16_15_08_12_id000--s22161/extra_snapshot_itr0.cpkl'
    # buffer_data = cloudpickle.load(
    #     open(buffer_path, 'rb'))
    # replay_buffer = buffer_data['replay_buffer']
    prefix = 'all_'

    path = Path(args.file)
    if path.is_file():
        extra_snapshot_path = path
        snapshot_dir = path.parent
    else:
        snapshot_dir = path
        extra_snapshot_path = snapshot_dir / 'extra_snapshot.cpkl'
    buffer_data = cloudpickle.load(
        open(extra_snapshot_path, 'rb'))
    if 'algorithm' in buffer_data:
        algorithm = buffer_data['algorithm']
        rl_replay_buffer = algorithm.replay_buffer
        enc_replay_buffer = algorithm.enc_replay_buffer
    else:
        rl_replay_buffer = buffer_data['replay_buffer']
        enc_replay_buffer = buffer_data['enc_replay_buffer']
    new_buffer = {
        'replay_buffer': rl_replay_buffer,
        'enc_replay_buffer': enc_replay_buffer,
    }
    save_file = snapshot_dir / 'encoder_and_rl_replay_buffer.joblib'
    if save_file.exists() and False:
        raise FileExistsError(str(save_file))
    else:
        # joblib.dump(new_buffer, str(save_file))
        # print('joblib saved to', save_file)
        with open(save_file, 'wb') as f:
            pickle.dump(new_buffer, f)
        print('pickle saved to', save_file)
    print('test')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=300,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=10,
                        help='Speedup')
    parser.add_argument('--mode', default='video_env', type=str,
                        help='env mode')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--pause', action='store_true')
    parser.add_argument('--enable_render', action='store_true')
    parser.add_argument('--multitaskpause', action='store_true')
    parser.add_argument('--hide', action='store_true')
    args = parser.parse_args()

    extract(args)
