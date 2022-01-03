import argparse
import time
import os.path as osp
import os

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.obs_dict_replay_buffer import ObsDictReplayBuffer
from rlkit.data_management.multitask_replay_buffer import ObsDictMultiTaskReplayBuffer
from rlkit.misc.wx250_utils import (add_multitask_data_to_singletask_buffer_real_robot, 
    add_multitask_data_to_multitask_buffer_real_robot, DummyEnv)
from rlkit.torch.task_encoders.encoder_decoder_nets import EncoderDecoderNet, TransformerEncoderDecoderNet

# import roboverse
import numpy as np
from pathlib import Path
import torch
import pickle

from rlkit.launchers.config import LOCAL_LOG_DIR

def get_buffer_size(data):
    num_transitions = 0
    for i in range(len(data)):
        for j in range(len(data[i]['observations'])):
            num_transitions += 1
    return num_transitions


def experiment(variant):
    image_size = 64
    if variant['task_encoder_checkpoint'] == '':
        task_embedding_dim = len(variant['buffers'])
    else:
        task_embedding_dim = variant['task_encoder_latent_dim']
    num_tasks = len(variant['buffers'])
    eval_env = DummyEnv(image_size=image_size, use_wrist=True, task_embedding_dim=task_embedding_dim)
    expl_env = eval_env
    action_dim = eval_env.action_space.low.size

    if variant['use_robot_state']:
        observation_keys = ['image', 'state', 'task_embedding']
        state_observation_dim = eval_env.observation_space.spaces['state'].low.size
    else:
        observation_keys = ['image', 'task_embedding']
        state_observation_dim = 0

    if variant['task_encoder_checkpoint'] != "":
        if variant['task_encoder_type'] == 'image':
            net = EncoderDecoderNet(64, task_embedding_dim, encoder_resnet=variant['use_task_encoder_resnet'])
        elif variant['task_encoder_type'] == 'trajectory':
            net = TransformerEncoderDecoderNet(64, task_embedding_dim, encoder_keys=variant['transformer_encoder_keys'])
        net.load_state_dict(torch.load(variant['task_encoder_checkpoint']))
        net.to(ptu.device)
        task_encoder = net.encoder_net    
    else:
        task_encoder = None

    replay_buffer = ObsDictMultiTaskReplayBuffer(
        int(1E6),
        expl_env,
        np.arange(num_tasks),
        use_next_obs_in_context=variant['use_next_obs_in_context'],
        sparse_rewards=False,
        observation_keys=observation_keys
    )
    buffer_params = {task: b for task, b in enumerate(variant['buffers'])}
    add_multitask_data_to_multitask_buffer_real_robot(buffer_params, replay_buffer, 
            task_encoder=task_encoder, embedding_mode=variant['embedding_mode'],
            encoder_type=variant['task_encoder_type'])

    if variant['use_negative_rewards']:
        if set(np.unique(replay_buffer._rewards)).issubset({0, 1}):
            replay_buffer._rewards = replay_buffer._rewards - 1.0
        assert set(np.unique(replay_buffer._rewards)).issubset({0, -1})

    save_file_name = "{}_real_robot_buffer.pkl".format(time.strftime("%y-%m-%d"))

    import gzip, pickle, pickletools
    with gzip.open(save_file_name, "wb") as f:
        pickled = pickle.dump(replay_buffer, f)
        optimized_pickle = pickletools.optimize(pickled)
        f.write(optimized_pickle)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffers", type=str, nargs='+')
    parser.add_argument("--buffer-variant", type=str, default="")
    parser.add_argument('--use-robot-state', action='store_true', default=False)
    parser.add_argument('--use-negative-rewards', action='store_true',
                        default=False)
    parser.add_argument("--task-encoder", default="", type=str)
    parser.add_argument("--encoder-type", default='image', choices=('image', 'trajectory'))
    parser.add_argument("--embedding-mode", type=str, choices=('one-hot', 'single', 'batch'), required=True)
    args = parser.parse_args()

    assert (args.embedding_mode == 'one-hot') ^ (args.task_encoder != "")
    assert args.buffer_variant != "" or len(args.buffers) > 0

    buffers = None
    if args.buffer_variant:
        '''Use buffer variant to get ordered list of buffer paths'''
        import json
        buffer_variant = open(args.buffer_variant)
        data = json.load(buffer_variant)
        buffers = data["buffers"]
    else:
        buffers = set()
        for buffer_path in args.buffers:
            if '.pkl' in buffer_path or '.npy' in buffer_path:
                buffers.add(buffer_path)
            else:
                path = Path(buffer_path)
                buffers.update(list(path.rglob('*.pkl')))
                buffers.update(list(path.rglob('*.npy')))
        buffers = [str(b) for b in buffers]
    print("Buffers Found:", buffers)

    variant = dict(
        num_epochs=3000,
        batch_size=64,
        meta_batch_size=4,
        max_path_length=25,
        num_trains_per_train_loop=1000,
        num_eval_steps_per_epoch=0,
        num_expl_steps_per_train_loop=0,
        min_num_steps_before_training=0,

        buffers=buffers,
        use_negative_rewards=args.use_negative_rewards,
        use_robot_state=args.use_robot_state,

        task_encoder_checkpoint=args.task_encoder,
        task_encoder_type=args.encoder_type,
        task_encoder_latent_dim=2,
        use_task_encoder_resnet=False,
        embedding_mode=args.embedding_mode,
        use_next_obs_in_context=False,
        transformer_encoder_keys=['observations']
    )

    experiment(variant)

