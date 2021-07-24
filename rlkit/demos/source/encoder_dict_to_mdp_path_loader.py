from collections import OrderedDict
import pickle
import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn
import torch.nn.functional as F
import copy
import rlkit.torch.pytorch_util as ptu
from rlkit.util.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer
from rlkit.demos.source.dict_to_mdp_path_loader import DictToMDPPathLoader
from rlkit.util.asset_loader import (
    load_local_or_remote_file, sync_down_folder, get_absolute_path, sync_down
)

import random
from rlkit.torch.core import np_to_pytorch_batch
from rlkit.data_management.path_builder import PathBuilder
from rlkit.launchers.config import LOCAL_LOG_DIR, AWS_S3_PATH
from rlkit.core import logger
import glob


class EncoderDictToMDPPathLoader(DictToMDPPathLoader):

    def __init__(
            self,
            trainer,
            replay_buffer,
            demo_train_buffer,
            demo_test_buffer,
            model=None,
            model_path=None,
            reward_fn=None,
            env=None,
            demo_paths=[], # list of dicts
            normalize=False,
            demo_train_split=0.9,
            demo_data_split=1,
            add_demos_to_replay_buffer=True,
            condition_encoding=False,
            bc_num_pretrain_steps=0,
            bc_batch_size=64,
            bc_weight=1.0,
            rl_weight=1.0,
            q_num_pretrain_steps=0,
            weight_decay=0,
            eval_policy=None,
            recompute_reward=False,
            object_list=None,
            env_info_key=None,
            obs_key=None,
            load_terminals=True,
            delete_after_loading=False,
            data_filter_fn=lambda x: True, # Return true to add path, false to ignore it
            **kwargs
    ):
        super().__init__(trainer,
            replay_buffer,
            demo_train_buffer,
            demo_test_buffer,
            demo_paths,
            demo_train_split,
            demo_data_split,
            add_demos_to_replay_buffer,
            bc_num_pretrain_steps,
            bc_batch_size,
            bc_weight,
            rl_weight,
            q_num_pretrain_steps,
            weight_decay,
            eval_policy,
            recompute_reward,
            env_info_key,
            obs_key,
            load_terminals,
            delete_after_loading,
            data_filter_fn,
            **kwargs)
       
        if model is None:
            self.model = load_local_or_remote_file(model_path, delete_after_loading=delete_after_loading)
        else:
            self.model = model
        self.condition_encoding = condition_encoding
        self.reward_fn = reward_fn
        self.normalize = normalize
        self.object_list = object_list
        self.env = env

    def preprocess(self, observation):
        observation = copy.deepcopy(observation)
        images = np.stack([observation[i]['image_observation'] for i in range(len(observation))])

        if self.normalize:
            images = images / 255.0

        if self.condition_encoding:
            cond = images[0].repeat(len(observation), axis=0)
            latents = self.model.encode_np(images, cond)
        else:
            latents = self.model.encode_np(images)

        for i in range(len(observation)):
            observation[i]["initial_latent_state"] = latents[0]
            observation[i]["latent_observation"] = latents[i]
            observation[i]["latent_achieved_goal"] = latents[i]
            observation[i]["latent_desired_goal"] = latents[-1]
            del observation[i]['image_observation']

        return observation

    def preprocess_array_obs(self, observation):
        new_observations = []
        for i in range(len(observation)):
            new_observations.append(dict(observation=observation[i]))
        return new_observations

    def encode(self, obs):
        if self.normalize:
            return ptu.get_numpy(self.model.encode(ptu.from_numpy(obs) / 255.0))
        return ptu.get_numpy(self.model.encode(ptu.from_numpy(obs)))


    def load_path(self, path, replay_buffer, obs_dict=None):
        # Filter data #
        if not self.data_filter_fn(path): return

        rewards = []
        path_builder = PathBuilder()

        H = min(len(path["observations"]), len(path["actions"]))
        if obs_dict:
            traj_obs = self.preprocess(path["observations"])
            next_traj_obs = self.preprocess(path["next_observations"])
        else:
            traj_obs = self.preprocess_array_obs(path["observations"])
            next_traj_obs = self.preprocess_array_obs(path["next_observations"])

        for i in range(H):
            ob = traj_obs[i]
            next_ob = next_traj_obs[i]
            action = path["actions"][i]
            reward = path["rewards"][i]
            terminal = path["terminals"][i]
            if not self.load_terminals:
                terminal = np.zeros(terminal.shape)
            agent_info = path["agent_infos"][i]
            env_info = path["env_infos"][i]
            if self.recompute_reward:
                reward = self.reward_fn(ob, action, next_ob, next_ob)

            reward = np.array([reward]).flatten()
            rewards.append(reward)
            terminal = np.array([terminal]).reshape((1,))
            path_builder.add_all(
                observations=ob,
                actions=action,
                rewards=reward,
                next_observations=next_ob,
                terminals=terminal,
                agent_infos=agent_info,
                env_infos=env_info,
            )
        self.demo_trajectory_rewards.append(rewards)
        path = path_builder.get_all_stacked()
        replay_buffer.add_path(path)
        print("rewards", np.min(rewards), np.max(rewards))
        print("loading path, length", len(path["observations"]), len(path["actions"]))
        print("actions", np.min(path["actions"]), np.max(path["actions"]))
        print("path sum rewards", sum(rewards), len(rewards))


class DualEncoderDictToMDPPathLoader(DictToMDPPathLoader):

    def __init__(
            self,
            trainer,
            replay_buffer,
            demo_train_buffer,
            demo_test_buffer,
            model=None,
            model_path=None,
            input_model=None,
            input_model_path=None,
            reward_fn=None,
            env=None,
            demo_paths=[], # list of dicts
            normalize=False,
            demo_train_split=0.9,
            demo_data_split=1,
            add_demos_to_replay_buffer=True,
            condition_input_encoding=False,
            bc_num_pretrain_steps=0,
            bc_batch_size=64,
            bc_weight=1.0,
            rl_weight=1.0,
            q_num_pretrain_steps=0,
            weight_decay=0,
            eval_policy=None,
            recompute_reward=False,
            object_list=None,
            env_info_key=None,
            obs_key=None,
            load_terminals=True,
            delete_after_loading=False,
            data_filter_fn=lambda x: True, # Return true to add path, false to ignore it
            **kwargs
    ):
        super().__init__(trainer,
            replay_buffer,
            demo_train_buffer,
            demo_test_buffer,
            demo_paths,
            demo_train_split,
            demo_data_split,
            add_demos_to_replay_buffer,
            bc_num_pretrain_steps,
            bc_batch_size,
            bc_weight,
            rl_weight,
            q_num_pretrain_steps,
            weight_decay,
            eval_policy,
            recompute_reward,
            env_info_key,
            obs_key,
            load_terminals,
            delete_after_loading,
            data_filter_fn,
            **kwargs)
       
        if model is None:
            self.model = load_local_or_remote_file(model_path, delete_after_loading=delete_after_loading)
        else:
            self.model = model

        if input_model is None:
            self.input_model = load_local_or_remote_file(input_model_path, delete_after_loading=delete_after_loading)
        else:
            self.input_model = input_model
        self.condition_input_encoding = condition_input_encoding
        self.reward_fn = reward_fn
        self.normalize = normalize
        self.object_list = object_list
        self.env = env

    def preprocess(self, observation):
        observation = copy.deepcopy(observation)
        images = np.stack([observation[i]['image_observation'] for i in range(len(observation))])

        if self.normalize:
            images = images / 255.0

        if self.condition_input_encoding:
            cond = images[0].repeat(len(observation), axis=0)
            input_latents = self.input_model.encode_np(images, cond)
        else:
            input_latents = self.input_model.encode_np(images)

        latents = self.model.encode_np(images)

        for i in range(len(observation)):
            observation[i]["initial_latent_state"] = latents[0]
            observation[i]["latent_observation"] = latents[i]
            observation[i]["latent_achieved_goal"] = latents[i]
            observation[i]["input_latent"] = input_latents[i]
            observation[i]["latent_desired_goal"] = latents[-1]
            del observation[i]['image_observation']

        return observation

    def preprocess_array_obs(self, observation):
        new_observations = []
        for i in range(len(observation)):
            new_observations.append(dict(observation=observation[i]))
        return new_observations

    def encode(self, obs):
        if self.normalize:
            return ptu.get_numpy(self.model.encode(ptu.from_numpy(obs) / 255.0))
        return ptu.get_numpy(self.model.encode(ptu.from_numpy(obs)))


    def load_path(self, path, replay_buffer, obs_dict=None):
        # Filter data #
        if not self.data_filter_fn(path): return

        rewards = []
        path_builder = PathBuilder()

        H = min(len(path["observations"]), len(path["actions"]))
        if obs_dict:
            traj_obs = self.preprocess(path["observations"])
            next_traj_obs = self.preprocess(path["next_observations"])
        else:
            traj_obs = self.preprocess_array_obs(path["observations"])
            next_traj_obs = self.preprocess_array_obs(path["next_observations"])

        for i in range(H):
            ob = traj_obs[i]
            next_ob = next_traj_obs[i]
            action = path["actions"][i]
            reward = path["rewards"][i]
            terminal = path["terminals"][i]
            if not self.load_terminals:
                terminal = np.zeros(terminal.shape)
            agent_info = path["agent_infos"][i]
            env_info = path["env_infos"][i]
            if self.recompute_reward:
                reward = self.reward_fn(ob, action, next_ob, next_ob)

            reward = np.array([reward]).flatten()
            rewards.append(reward)
            terminal = np.array([terminal]).reshape((1,))
            path_builder.add_all(
                observations=ob,
                actions=action,
                rewards=reward,
                next_observations=next_ob,
                terminals=terminal,
                agent_infos=agent_info,
                env_infos=env_info,
            )
        self.demo_trajectory_rewards.append(rewards)
        path = path_builder.get_all_stacked()
        replay_buffer.add_path(path)
        print("rewards", np.min(rewards), np.max(rewards))
        print("loading path, length", len(path["observations"]), len(path["actions"]))
        print("actions", np.min(path["actions"]), np.max(path["actions"]))
        print("path sum rewards", sum(rewards), len(rewards))