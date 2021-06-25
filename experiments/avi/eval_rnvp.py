import argparse
import os
import torch

from train_latent_intention_policy import enable_gpus
import rlkit.torch.pytorch_util as ptu
from rlkit.misc.roboverse_utils import dump_video_basic
import roboverse

CHECKPOINT = '/media/avi/data/Work/doodad_output/21-06-23-rnvp-robot/21-06-23-rnvp-robot_2021_06_23_10_28_18_id000--s0/itr_0.pt'

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT)
    parser.add_argument("--gpu", default='0', type=str)
    args = parser.parse_args()

    enable_gpus(args.gpu)
    ptu.set_gpu_mode(True)
    checkpoint_dir = os.path.dirname(args.checkpoint)
    # json_path = os.path.join(checkpoint_dir, 'variant.json')
    # with open(json_path, 'r') as f:
    #     variant = json.load(f)

    with open(args.checkpoint, 'rb') as f:
        params = torch.load(f)
        # params = pickle.load(f)

    rnvp_policy = params['trainer/bijector']
    rnvp_policy.eval()

    variant = dict(env='Widow250PickPlaceMedium-v0')

    env = roboverse.make(variant['env'], transpose_image=True)
    # env = roboverse.make(variant['env'], transpose_image=True, gui=True)
    paths = []
    env.reset()

    for _ in range(10):
        # x = np.random.randn(8,)
        next_observations = []
        env.reset_robot_only()
        for _ in range(35):
            obs = env.get_observation()
            next_observations.append(obs)
            # action, _ = rnvp_policy.get_action(obs['image'], x)
            action, _ = rnvp_policy.get_action(obs['image'])
            obs, rew, done, info = env.step(action)

        paths.append(dict(next_observations=next_observations))

    video_dir = '/media/avi/data/Work/doodad_output/test_rnvp_videos/chkpt_300_fixed'
    dump_video_basic(video_dir, paths)
