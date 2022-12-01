from rlkit.envs.wrappers.normalized_box_env import NormalizedBoxEnv
from iris_robots.robot_env import RobotEnv

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
    def __init__(self, init_obs, normalize=False):
        self._previous_obs = init_obs
        self._obs = init_obs
        
        from sklearn import linear_model
        from sklearn.metrics import mean_squared_error
        import pickle

        self.model_path = '/iris/u/jyang27/dev/iris_robots/widowx_scripts/linear_cdp_model.pkl'
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)

        if normalize:
            self.normalization_path = '/iris/u/jyang27/dev/iris_robots/widowx_scripts/action_normalization_mean.pkl' 
            with open(self.normalization_path, 'rb') as f:
                self.x_mean, self.x_std, self.y_mean, self.y_std = pickle.load(f)  
        else:
            x_mean, x_std = np.mean(x_train, axis=0), np.std(x_train, axis=0)
            y_mean, y_std = np.mean(y_train, axis=0), np.std(y_train, axis=0)
    
 
    def postprocess_obs_action(self, obs, action):
        self._previous_obs = self._obs
        self._obs = obs
        adp = action.tolist()
        adp += self._obs['current_pose'].tolist()
        adp += self._obs['desired_pose'].tolist()
        adp += self._previous_obs['current_pose'].tolist()
        adp += self._previous_obs['desired_pose'].tolist()
        adp = np.array(adp).reshape(1, -1)
        return self.model.predict((adp - self.x_mean)/self.x_std)[0]*self.y_std + self.y_mean


def process_image(image, downsample=False):
    ''' ObsDictReplayBuffer wants flattened (channel, height, width) images float32'''
    if image.dtype == np.uint8:
        image =  image.astype(np.float32) / 255.0
    if len(image.shape) == 3 and image.shape[0] != 3 and image.shape[2] == 3:
        image = np.transpose(image, (2, 0, 1))
    if downsample:
        image = image[:,::2, ::2]
    return image.flatten()

def process_obs(obs, task, use_robot_state, prev_obs=None, downsample=False):
    if use_robot_state:
        observation_keys = ['image', 'desired_pose', 'current_pose','joint_positions', 'joint_velocities', 'task_embedding']
    else:
        observation_keys = ['image', 'task_embedding']

    if prev_obs:
        observation_keys = ['previous_image'] + observation_keys

    if task is None:
        observation_keys = observation_keys[:-1]

    obs['image'] = process_image(obs['images'][0]['array'], downsample=downsample)
    if prev_obs is not None:
        obs['previous_image'] = process_image(prev_obs['images'][0]['array'], downsample=downsample)
    obs['task_embedding'] = task
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
    parser.add_argument("--task-encoder", default=None)
    parser.add_argument("--sample-trajectory", type=str, default=None)
    parser.add_argument("--use-checkpoint-encoder", action='store_true', default=False)
    parser.add_argument("--use-robot-state", action='store_true', default=False)
    parser.add_argument("--achieved-action-relabelling", action="store_true", default=False)
    parser.add_argument("--normalize-relabelling", action="store_true", default=False)
    parser.add_argument("--robot-model", type=str, choices=('wx250s', 'franka'), default='wx250s')
    parser.add_argument("--stack-frames", action='store_true', default=False)
    parser.add_argument("--downsample-image", action='store_true', default=False)
    args = parser.parse_args()

    if not os.path.exists(args.video_save_dir) and args.video_save_dir:
        os.mkdir(args.video_save_dir)

    ptu.set_gpu_mode(True)

    if args.robot_model == 'wx250s':
    	env = RobotEnv(robot_model='wx250s', control_hz=20, use_local_cameras=True, camera_types='cv2', blocking=False)
    else:
        env = RobotEnv('172.16.0.21', use_robot_cameras=True)
    env.reset()

    checkpoint_path = args.checkpoint_path
    _, ext = os.path.splitext(args.checkpoint_path)

    with open(args.checkpoint_path, 'rb') as handle:
        if ext == ".pt":
            params = torch.load(handle)
            if args.q_value_eval:
                eval_policy = max_q_policy(params['trainer/qf1'], params['trainer/policy'])
            else:
                eval_policy = params['evaluation/policy']
            if args.use_checkpoint_encoder:
                task_encoder = params['trainer/task_encoder']
                task_encoder.to(ptu.device)
        elif ext == ".pkl":
            eval_policy = pickle.load(handle)

    eval_policy.eval()
    eval_policy.color_jitter = False    
    try:
        norm =  eval_policy.feature_norm 
    except:
        eval_policy.feature_norm = False

    if args.task_encoder:
        from rlkit.torch.task_encoders.encoder_decoder_nets import EncoderDecoderNet
        net = EncoderDecoderNet(64, 2, encoder_resnet=False)
        net.load_state_dict(torch.load(args.task_encoder))
        net.to(ptu.device)
        task_encoder = net.encoder_net

    for i in range(num_trajs):
        obs = env.reset()
        if args.achieved_action_relabelling:
            action_postprocessor = DeltaPoseToCommand(obs, normalize=args.normalize_relabelling)

        images = []

        if not args.task_embedding:
            if args.num_tasks != 0:
                valid_task_idx = False
                while not valid_task_idx:
                    task_idx = "None"
                    while not task_idx.isnumeric():
                        task_idx = input("Enter task idx to continue...")
                    task_idx = int(task_idx)
                    valid_task_idx = task_idx in list(range(args.num_tasks))
                task = np.array([0] * args.num_tasks)
                task[task_idx] = 1
            else:
                task = None
        else:
            if args.task_encoder or args.use_checkpoint_encoder:
                if args.sample_trajectory is None:
                    env.move_to_state([-0.11982477,  0.2200,  0.07], 0, duration=1)
                    input("Press enter to take image")
                    obs = ptu.from_numpy(env.get_observation()['image'].reshape(1, 1, -1))
                else:
                    with open(args.sample_trajectory, 'rb') as f:
                        path = pickle.load(f)
                    obs = ptu.from_numpy(path['observations'][-1]['image'].reshape(1, 1, -1))
                task = ptu.get_numpy(task_encoder(obs[0]))
                task = task.reshape(-1)
                print("task: ", task)
                input("Press enter to continue")
                obs = env.reset()
            else:
                task = "None"
                while not isinstance(eval(task), list):
                    task = input("Enter task embedding to continue...")
                task = np.array(eval(task))
        print("Eval Traj {}".format(i))


        if args.stack_frames:
            prev_obs = obs
        if args.data_save_dir is not None:
            trajectory = []

        for j in range(args.num_timesteps):
            obs = env.get_observation()
            if args.stack_frames:
                obs_flat = process_obs(obs, task, args.use_robot_state, prev_obs=prev_obs,
                        downsample=args.downsample_image)
            else:
                obs_flat = process_obs(obs, task, args.use_robot_state,
                            downsample=args.downsample_image)
            action, info = eval_policy.get_action(obs_flat)

            if args.data_save_dir is not None:
                transition = [obs, action]
            
            if args.achieved_action_relabelling:
                action = action_postprocessor.postprocess_obs_action(obs, action)

            if args.data_save_dir is not None:
                transition.append(action)
                trajectory.append(transition)

            action = process_action(action)
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
