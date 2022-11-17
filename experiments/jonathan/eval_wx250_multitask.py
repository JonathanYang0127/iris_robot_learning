from rlkit.envs.wrappers.normalized_box_env import NormalizedBoxEnv
from iris_robots.robot_env import RobotEnv

import argparse
import os
import pickle
import torch
import numpy as np
from PIL import Image

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
    def __init__(self, init_obs):
        self._previous_obs = init_obs
        self._obs = obs
        
        from sklearn import linear_model
        from sklearn.metrics import mean_squared_error
        import pickle

        self.model_path = '/iris/u/jyang27/dev/iris_robots/widowx_scripts/linear_cdp_model.pkl'
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        
    def postprocess_obs_action(self, obs, action):
        self._previous_obs = self._obs
        self._obs = obs
        adp = action.tolist()
        adp += self._obs['current_pose'].tolist()
        adp += self._obs['desired_pose'].tolist()
        adp += self._previous_obs['current_pose'].tolist()
        adp += self._previous_obs['desired_pose'].tolist()
        adp = np.array(adp).reshape(1, -1)
        return self.model.predict(adp)[0]


def process_image(image):
    ''' ObsDictReplayBuffer wants flattened (channel, height, width) images float32'''
    if image.dtype == np.uint8:
        image =  image.astype(np.float32) / 255.0
    if len(image.shape) == 3 and image.shape[0] != 3 and image.shape[2] == 3:
        image = np.transpose(image, (2, 0, 1))
        image = image.flatten()
    return image

def process_obs(obs, task, use_robot_state):
    if use_robot_state:
        observation_keys = ['image', 'desired_pose', 'current_pose','joint_positions', 'joint_velocities', 'task_embedding']
    else:
        observation_keys = ['image', 'task_embedding']
    obs['image'] = process_image(obs['images'][0]['array'])
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
    parser.add_argument("-n", "--num-timesteps", type=int, default=15)
    parser.add_argument("--q-value-eval", default=False, action='store_true')
    parser.add_argument("--num-tasks", type=int, default=0)
    parser.add_argument("--task-embedding", default=False, action="store_true")
    parser.add_argument("--task-encoder", default=None)
    parser.add_argument("--sample-trajectory", type=str, default=None)
    parser.add_argument("--use-checkpoint-encoder", action='store_true', default=False)
    parser.add_argument("--use-robot-state", action='store_true', default=False)
    parser.add_argument("--achieved-action-relabelling", action="store_true", default=False)
    parser.add_argument("--robot-model", type=str, choices=('wx250s', 'franka'), default='wx250s')
    args = parser.parse_args()

    assert args.num_tasks != 0 or args.task_embedding
    if not os.path.exists(args.video_save_dir) and args.video_save_dir:
        os.mkdir(args.video_save_dir)

    ptu.set_gpu_mode(True)

    if args.robot_model == 'wx250s':
    	env = RobotEnv(robot_model='wx250s', control_hz=20, use_local_cameras=True, camera_types='cv2', blocking=False)
    else:
        env = RobotEnv('172.16.0.21', use_local_cameras=True)
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

    if args.task_encoder:
        from rlkit.torch.task_encoders.encoder_decoder_nets import EncoderDecoderNet
        net = EncoderDecoderNet(64, 2, encoder_resnet=False)
        net.load_state_dict(torch.load(args.task_encoder))
        net.to(ptu.device)
        task_encoder = net.encoder_net

    for i in range(num_trajs):
        obs = env.reset()
        if args.achieved_action_relabelling:
            action_postprocessor = DeltaPoseToCommand(obs)

        images = []

        if not args.task_embedding:
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


        for j in range(args.num_timesteps):
            obs = env.get_observation()
            obs_flat = process_obs(obs, task, args.use_robot_state)
            action, info = eval_policy.get_action(obs_flat)
            if args.achieved_action_relabelling:
                action = action_postprocessor.postprocess_obs_action(obs, action)
            action = process_action(action)
            env.step(action)

            if args.video_save_dir:
                if full_image:
                    image = obs['full_image']
                else:
                    image = np.transpose(obs['images'][0]['array'], (1, 2, 0))
                images.append(Image.fromarray(image))

        # Save Video
        if args.video_save_dir:
            print("Saving Video")
            images[0].save('{}/eval_{}.gif'.format(args.video_save_dir, i),
                            format='GIF', append_images=images[1:],
                            save_all=True, duration=200, loop=0)
