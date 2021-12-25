import argparse
import os
import numpy as np
import pickle as pkl
from tqdm import tqdm

import roboverse

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str)
parser.add_argument("--num_trajectories", type=int, default=100)
parser.add_argument("--num_timesteps", type=int, default=50)
parser.add_argument("--downsample", action='store_true')
parser.add_argument("--test_env_seeds", nargs='+', type=int)
parser.add_argument("--gui", dest="gui", action="store_true", default=False)
parser.add_argument("--video_save_frequency", type=int,
                    default=0, help="Set to zero for no video saving")

# Environment arguments.
parser.add_argument('--drawer_sliding', action='store_true')
parser.add_argument('--fix_drawer_orientation', action='store_true')
parser.add_argument('--fix_drawer_orientation_semicircle', action='store_true')
parser.add_argument('--new_view', action='store_true')
parser.add_argument('--close_view', action='store_true')
parser.add_argument('--red_drawer_base', action='store_true')

args = parser.parse_args()

# data_save_path = "/2tb/home/patrickhaoy/data/affordances/data/reset_free_v5_rotated_top_drawer/top_drawer_goals.pkl"  # NOQA


def presample_goal(args, test_env_seed):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    kwargs = {
        'drawer_sliding': args.drawer_sliding,
        'fix_drawer_orientation': args.fix_drawer_orientation,
        'fix_drawer_orientation_semicircle': (
            args.fix_drawer_orientation_semicircle),
        'new_view': args.new_view,
        'close_view': args.close_view,
        'red_drawer_base': args.red_drawer_base,
    }

    if test_env_seed is None:
        output_path = os.path.join(args.output_dir, 'goals.pkl')
    else:
        output_path = os.path.join(args.output_dir,
                                   'goals_seed%d.pkl' % (test_env_seed))
        kwargs['test_env_seed'] = test_env_seed

    if args.downsample:
        kwargs['downsample'] = True
        kwargs['env_obs_img_dim'] = 196

    env = roboverse.make('SawyerRigAffordances-v1', test_env=True, **kwargs)

    obs_dim = env.observation_space.spaces['state_achieved_goal'].low.size
    imlength = env.obs_img_dim * env.obs_img_dim * 3

    dataset = {
        'initial_latent_state': np.zeros(
            (args.num_trajectories * args.num_timesteps, 720),
            dtype=np.float),
        'latent_desired_goal': np.zeros(
            (args.num_trajectories * args.num_timesteps, 720),
            dtype=np.float),
        'state_desired_goal': np.zeros(
            (args.num_trajectories * args.num_timesteps, obs_dim),
            dtype=np.float),
        'image_desired_goal': np.zeros(
            (args.num_trajectories * args.num_timesteps, imlength),
            dtype=np.float),
        'initial_image_observation': np.zeros(
            (args.num_trajectories * args.num_timesteps, imlength),
            dtype=np.float),
    }

    for i in tqdm(range(args.num_trajectories)):
        env.demo_reset()
        init_img = np.uint8(env.render_obs()).transpose() / 255.0

        for _ in range(40):
            action = env.get_demo_action()
            obs, reward, done, info = env.step(action)

        for t in range(args.num_timesteps):
            action = env.get_demo_action()
            obs, reward, done, info = env.step(action)

            img = np.uint8(env.render_obs()).transpose() / 255.0

            ind = i * args.num_timesteps + t
            dataset['initial_image_observation'][ind] = init_img.flatten()
            dataset['state_desired_goal'][ind] = obs['state_achieved_goal']
            dataset['image_desired_goal'][ind] = img.flatten()

    file = open(output_path, 'wb')
    pkl.dump(dataset, file)
    file.close()


test_env_seeds = args.test_env_seeds

if test_env_seeds is None:
    presample_goal(args, None)
else:
    for test_env_seed in test_env_seeds:
        presample_goal(args, test_env_seed)
