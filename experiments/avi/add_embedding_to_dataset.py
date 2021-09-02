import numpy as np
import torch
import os

from rlkit.data_management.multitask_replay_buffer import ObsDictMultiTaskReplayBuffer
import rlkit.torch.pytorch_util as ptu
import roboverse
from rlkit.misc.roboverse_utils import add_reward_filtered_data_to_buffers_multitask, \
    get_buffer_size_multitask, get_buffer_size
from rlkit.torch.task_encoders.encoder_decoder_nets import EncoderDecoderNet

import argparse
from tqdm import tqdm

ENV = 'Widow250PickPlaceMetaTestMultiObjectMultiContainer-v0'
# BUFFER = '/nfs/kun1/users/avi/scripted_sim_datasets/june30_validation_Widow250PickPlaceMetaTrainMultiObjectMultiContainer-v0_3K_save_all_noise_0.1_2021-06-30T15-52-04/june30_validation_Widow250PickPlaceMetaTrainMultiObjectMultiContainer-v0_3K_save_all_noise_0.1_2021-06-30T15-52-04_3200.npy'
BUFFER = '/media/avi/data/Work/github/avisingh599/minibullet/data/jul3_Widow250PickPlaceMetaTrainMultiObjectMultiContainer-v0_1000_save_all_noise_0.1_2021-07-03T09-00-06/jul3_Widow250PickPlaceMetaTrainMultiObjectMultiContainer-v0_1000_save_all_noise_0.1_2021-07-03T09-00-06_1000.npy'
# CHECKPOINT = '/nfs/kun1/users/avi/doodad-output/21-07-02-task-encoder-decoder/21-07-02-task-encoder-decoder_2021_07_02_13_07_33_id000--s0/itr_4990.pt'
CHECKPOINT = '/home/avi/Downloads/itr_4990.pt'


def main(args):

    with open(args.buffer, 'rb') as fl:
        data = np.load(fl, allow_pickle=True)

    output_filename = 'embedding_' + os.path.basename(args.buffer)
    save_path = os.path.join(os.path.dirname(args.buffer), output_filename)
    num_transitions = get_buffer_size_multitask(data)
    max_replay_buffer_size = num_transitions + 10
    expl_env = roboverse.make(ENV, transpose_image=True)
    observation_keys = ['image']
    train_task_indices = list(range(32))

    replay_buffer_positive_val = ObsDictMultiTaskReplayBuffer(
        int(max_replay_buffer_size/2),
        expl_env,
        train_task_indices,
        use_next_obs_in_context=False,
        sparse_rewards=False,
        observation_keys=observation_keys
    )

    replay_buffer_full_val = ObsDictMultiTaskReplayBuffer(
        max_replay_buffer_size,
        expl_env,
        train_task_indices,
        use_next_obs_in_context=False,
        sparse_rewards=False,
        observation_keys=observation_keys
    )

    add_reward_filtered_data_to_buffers_multitask(data, observation_keys,
                                                  (replay_buffer_positive_val, lambda r: r > 0),
                                                  (replay_buffer_full_val, lambda r: True))

    latent_dim = 2
    net = EncoderDecoderNet(args.image_dim, latent_dim, encoder_resnet=False)
    net.load_state_dict(torch.load(args.chkpt))
    net.cuda()

    for j in tqdm(range(len(data))):
        assert (len(data[j]['actions']) == len(data[j]['observations']) == len(
            data[j]['next_observations']))

        task_idx = data[j]['env_infos'][0]['task_idx']
        tasks_to_sample = [task_idx]
        traj_len = len(data[j]['observations'])

        encoder_batch_val = replay_buffer_positive_val.sample_batch(tasks_to_sample, traj_len)
        decoder_batch_val = replay_buffer_full_val.sample_batch(tasks_to_sample, traj_len)
        reward_predictions, mu, logvar = net.forward(
            ptu.from_numpy(encoder_batch_val['observations']),
            ptu.from_numpy(decoder_batch_val['observations']))

        for i in range(traj_len):
            data[j]['observations'][i]['task_embedding'] = ptu.get_numpy(mu[i])
            data[j]['next_observations'][i]['task_embedding'] = ptu.get_numpy(mu[i])

    print('saving..')
    np.save(save_path, data)
    print(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--env", type=str, default='Widow250PickPlaceMetaTrainMultiObjectMultiContainer-v0')
    parser.add_argument("--buffer", type=str, default=BUFFER)
    parser.add_argument("--chkpt", type=str, default=CHECKPOINT)
    parser.add_argument("--num-tasks", default=2, type=int)
    parser.add_argument("--gpu", default='0', type=str)
    parser.add_argument("--image-dim", default=48, type=int)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    ptu.set_gpu_mode(True)

    main(args)
