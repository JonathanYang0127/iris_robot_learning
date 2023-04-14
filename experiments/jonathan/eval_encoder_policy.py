from rlkit.envs.wrappers.normalized_box_env import NormalizedBoxEnv
from iris_robots.robot_env import RobotEnv
from iris_robots.transformations import add_angles, angle_diff, pose_diff
from rlkit.torch.task_encoders.encoder_decoder_nets import TransformerEncoderDecoderNet

import argparse
import os
import pickle
import torch
import numpy as np
from PIL import Image
from datetime import datetime

import rlkit.torch.pytorch_util as ptu

class max_q_policy:
    def __init__(self, qf1, policy, use_robot_state=True, num_repeat=100):
        self.qf1 = qf1
        self.policy = policy
        self.use_robot_state = use_robot_state
        self.num_repeat = num_repeat

    def get_action(self, obs):
        """
        Used when sampling actions from the policy and doing max Q-learning
        """
        with torch.no_grad():
            obs = obs.view(1, -1).repeat(self.num_repeat, 1)
            action = self.policy(obs).rsample()
            q1 = self.qf1(obs, action)
            print(q1)
            ind = q1.max(0)[1]
        return ptu.get_numpy(action[ind]).flatten(), None


    def eval(self):
        self.qf1.eval()
        self.policy.eval()


class DeltaPoseToCommand:
    def __init__(self, init_obs, robot_type, normalize=False, model_type='nonlinear'):
        self._previous_obs = init_obs
        self._obs = init_obs
        self.model_type = model_type

        from sklearn import linear_model
        from sklearn.metrics import mean_squared_error
        import pickle

        if self.model_type == 'linear':
            self.model_path = '/iris/u/jyang27/dev/iris_robots/widowx_scripts/linear_cdp_model.pkl'
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
        elif self.model_type == 'nonlinear':
            self.model_path = '/iris/u/jyang27/dev/iris_robots/widowx_scripts/nonlinear_{}_adp_cdp_xyz_model_'.format(robot_type)
            self.angle_model_path =  '/iris/u/jyang27/dev/iris_robots/widowx_scripts/nonlinear_{}_adp_cdp_angle_model_'.format(robot_type)
            if normalize:
                self.model_path += 'normalized.pt'
                self.angle_model_path += 'normalized.pt'
            else:
                self.model_path += 'unnormalized.pt'
                self.angle_model_path += 'unnormalized.pt'
            with open(self.model_path, 'rb') as f:
                self.model = torch.load(f)
            with open(self.angle_model_path, 'rb') as f:
                self.angle_model = torch.load(f)

        self.normalization_path_xyz = '/iris/u/jyang27/dev/iris_robots/widowx_scripts/action_normalization_mean_adp_cdp_xyz.pkl'
        with open(self.normalization_path_xyz, 'rb') as f:
            self.x_mean_xyz, self.x_std_xyz, self.y_mean_xyz, self.y_std_xyz = pickle.load(f)

        self.normalization_path_angle = '/iris/u/jyang27/dev/iris_robots/widowx_scripts/action_normalization_mean_adp_cdp_angle.pkl'
        with open(self.normalization_path_angle, 'rb') as f:
            self.x_mean_angle, self.x_std_angle, self.y_mean_angle, self.y_std_angle = pickle.load(f)

        if not normalize:
            self.x_mean_xyz, self.x_std_xyz = np.zeros(self.x_mean_xyz.shape[0]), np.ones(self.x_std_xyz.shape[0])
            self.y_mean_xyz, self.y_std_xyz = np.zeros(self.y_mean_xyz.shape[0]), np.ones(self.y_std_xyz.shape[0])

            self.x_mean_angle, self.x_std_angle = np.zeros(self.x_mean_angle.shape[0]), np.ones(self.x_std_angle.shape[0])
            self.y_mean_angle, self.y_std_angle = np.zeros(self.y_mean_angle.shape[0]), np.ones(self.y_std_angle.shape[0])


    def set_init_obs(self, obs):
        self._previous_obs = obs
        self._obs = obs

    def postprocess_obs_action(self, obs, action):
        self._previous_obs = self._obs
        self._obs = obs
        adp = action.tolist()[:-1]
        adp += self._obs['current_pose'].tolist()[:-1]
        adp += self._obs['joint_positions'].tolist()
        #adp += self._obs['desired_pose'].tolist()[:-1]
        adp += self._previous_obs['current_pose'].tolist()[:-1]
        adp += self._previous_obs['joint_positions'].tolist()
        #adp += self._previous_obs['desired_pose'].tolist()[:-1]
        adp = np.array(adp).reshape(1, -1)

        adp = (adp - self.x_mean_xyz) / self.x_std_xyz
        if self.model_type == 'linear':
            return self.model.predict(adp)[0]*self.y_std + self.y_mean
        elif self.model_type == 'nonlinear':
            adp = torch.Tensor(adp).cuda()
            xyz = self.model(adp).detach().cpu().numpy()[0]*self.y_std_xyz + self.y_mean_xyz
            angle_repr = self.angle_model(adp).detach().cpu().numpy()[0]
            angle = np.arctan2(angle_repr[:3], angle_repr[3:6])
            return np.concatenate((xyz, angle))


def process_image(image, downsample=False):
    ''' ObsDictReplayBuffer wants flattened (channel, height, width) images float32'''
    if image.dtype == np.uint8:
        image =  image.astype(np.float32) / 255.0
    if len(image.shape) == 3 and image.shape[0] != 3 and image.shape[2] == 3:
        image = np.transpose(image, (2, 0, 1))
    if downsample:
        image = image[:,::2, ::2]
    return image.flatten()

def process_obs(obs, use_robot_state, prev_obs=None, downsample=False):
    if use_robot_state:
        observation_keys = ['image', 'desired_pose', 'current_pose']
    else:
        observation_keys = ['image']

    if prev_obs:
        observation_keys = ['previous_image'] + observation_keys

    if task is None:
        observation_keys = observation_keys[:-1]

    obs['image'] = process_image(obs['images'][0]['array'], downsample=downsample)
    if prev_obs is not None:
        obs['previous_image'] = process_image(prev_obs['images'][0]['array'], downsample=downsample)
    return ptu.from_numpy(np.concatenate([obs[k] for k in observation_keys]))

def process_action(action):
    return np.clip(action, -1, 1)


if __name__ == '__main__':
    num_trajs = 100
    full_image = True

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint-path", type=str, required=True)
    parser.add_argument("-v", "--video-save-dir", type=str, default="")
    parser.add_argument("-d", "--data-save-dir", type=str, default=None)
    parser.add_argument("-n", "--num-timesteps", type=int, default=15)
    parser.add_argument("--q-value-eval", default=False, action='store_true')
    parser.add_argument("--num-tasks", type=int, default=0)
    parser.add_argument("--task-embedding", default=False, action="store_true")
    parser.add_argument("--sample-trajectory", type=str, default=None)
    parser.add_argument("--use-checkpoint-encoder", action='store_true', default=False)
    parser.add_argument("--use-robot-state", action='store_true', default=False)
    parser.add_argument("--action-relabelling", type=str, choices=("achieved, cdp"))
    parser.add_argument("--normalize-relabelling", action="store_true", default=False)
    parser.add_argument("--robot-model", type=str, choices=('wx250s', 'franka'), default='wx250s')
    parser.add_argument("--stack-frames", action='store_true', default=False)
    parser.add_argument("--downsample-image", action='store_true', default=False)
    parser.add_argument("--multi-head-idx", type=int, default=-1)
    args = parser.parse_args()

    if not os.path.exists(args.video_save_dir) and args.video_save_dir:
        os.mkdir(args.video_save_dir)

    ptu.set_gpu_mode(True)

    if args.robot_model == 'wx250s':
    	env = RobotEnv(robot_model='wx250s', control_hz=20, use_local_cameras=True, camera_types='cv2', blocking=False)
    else:
        env = RobotEnv('172.16.0.21', use_robot_cameras=True)
    obs = env.reset()

    if args.action_relabelling == 'achieved':
        action_postprocessor = DeltaPoseToCommand(obs, args.robot_model, normalize=args.normalize_relabelling)

    checkpoint_path = args.checkpoint_path
    _, ext = os.path.splitext(args.checkpoint_path)

    with open(args.checkpoint_path, 'rb') as handle:
        params = torch.load(handle)
        from rlkit.torch.task_encoders.encoder_decoder_nets import TransformerEncoderDecoderNet
        
        net = TransformerEncoderDecoderNet(64, 2, 110,
            image_augmentation=False,
            encoder_keys=['observations'],
            decoder_output_dim=7,
            decoder_type='small'
        )
        net.load_state_dict(params)
        eval_policy = net.decoder_net.cuda()
        
    eval_policy.eval()


    for i in range(num_trajs):
        obs = env.reset()
        if args.action_relabelling == 'achieved':
            action_postprocessor.set_init_obs(obs)

        images = []
        task = np.array([[0, 0]])
        task = ptu.from_numpy(task)


        if args.stack_frames:
            prev_obs = obs
        if args.data_save_dir is not None:
            trajectory = []

        for j in range(args.num_timesteps):
            obs = env.get_observation()
            if args.stack_frames:
                obs_flat = process_obs(obs, args.use_robot_state, prev_obs=prev_obs,
                        downsample=args.downsample_image)
            else:
                obs_flat = process_obs(obs, args.use_robot_state,
                            downsample=args.downsample_image)
            if args.multi_head_idx != -1:
                dist = eval_policy(task, obs_flat.view(1, -1))[args.multi_head_idx]
                action = dist
                #action = dist.rsample()[0].detach().cpu().numpy()
            else:
                action = eval_policy(task, obs_flat.view(1, -1)).detach().cpu().numpy()[0]

            if args.data_save_dir is not None:
                transition = [obs, action]
            
            if args.action_relabelling == 'achieved':
                #gripper = action[6]

                gripper = 0
                action = action_postprocessor.postprocess_obs_action(obs, action)
                action = np.concatenate((action, [gripper]))

            if args.data_save_dir is not None:
                transition.append(action)
                trajectory.append(transition)

            if args.action_relabelling == 'cdp' or args.action_relabelling == 'achieved':
                #action = process_action(action)
                print(action)
                if args.robot_model == 'franka':
                    action[3:6] *= -1
                env.step_direct(action)
            else:
                action = process_action(action)
                if args.robot_model == 'franka':
                    action[3:6] *= -1
                env.step(action)

            if args.stack_frames:
                prev_obs = obs

            if args.video_save_dir:
                image = obs['images'][0]['array']
                images.append(Image.fromarray(image))

        #Save Trajectory
        if args.data_save_dir is not None:
            now = datetime.now()
            dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")

            with open(os.path.join(args.data_save_dir, 'traj_{}.pkl'.format(dt_string)), 'wb+') as f:
                pickle.dump(trajectory, f)

        # Save Video
        if args.video_save_dir:
            print("Saving Video")
            images[0].save('{}/eval_{}.gif'.format(args.video_save_dir, i),
                            format='GIF', append_images=images[1:],
                            save_all=True, duration=200, loop=0)
