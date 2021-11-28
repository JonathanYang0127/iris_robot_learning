from rlkit.envs.wrappers import StackObservationEnv, RewardWrapperEnv
import rlkit.torch.pytorch_util as ptu
from rlkit.samplers.data_collector.step_collector import MdpStepCollector
from rlkit.samplers.data_collector.path_collector import GoalConditionedPathCollector
from rlkit.torch.networks import ConcatMlp, Mlp
from rlkit.torch.networks.cnn import ConcatCNN, CNN
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.envs.images import EnvRenderer, InsertImageEnv
from rlkit.util.io import load_local_or_remote_file
from rlkit.torch.sac.awac_trainer import AWACTrainer
from rlkit.torch.sac.iql_trainer import IQLTrainer
from rlkit.torch.sac.policies import GaussianPolicy, GaussianMixturePolicy, GaussianCNNPolicy, GaussianTwoChannelCNNPolicy
from rlkit.torch.torch_rl_algorithm import (
    TorchBatchRLAlgorithm,
    TorchOnlineRLAlgorithm,
)

from rlkit.demos.source.mdp_path_loader import MDPPathLoader
from rlkit.visualization.video import save_paths
import numpy as np
import torch
from rlkit.visualization.video import save_paths, VideoSaveFunction, RIGVideoSaveFunction
from rlkit.envs.images import Renderer, InsertImageEnv, EnvRenderer
from rlkit.launchers.contextual.util import (
    get_save_video_function,
    get_gym_env,
)

from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.gaussian_and_epsilon import GaussianAndEpsilonStrategy
from rlkit.exploration_strategies.ou_strategy import OUStrategy

import os.path as osp
from rlkit.core import logger
from rlkit.util.io import load_local_or_remote_file
from rlkit.launchers.contextual.rig.rig_launcher import StateImageGoalDiagnosticsFn
from rlkit.data_management.obs_dict_replay_buffer import \
        ObsDictRelabelingBuffer
from rlkit.data_management.wrappers.concat_to_obs_wrapper import \
        ConcatToObsWrapper
from rlkit.envs.reward_mask_wrapper import DiscreteDistribution, RewardMaskWrapper

from functools import partial
import rlkit.samplers.rollout_functions as rf

from rlkit.envs.contextual import ContextualEnv
import pickle

from rlkit.envs.contextual.goal_conditioned import (
    GoalDictDistributionFromMultitaskEnv,
    ContextualRewardFnFromMultitaskEnv,
    AddImageDistribution,
    GoalConditionedDiagnosticsToContextualDiagnostics,
    IndexIntoAchievedGoal,
)


from torch.utils import data
from rlkit.data_management.contextual_replay_buffer import (
    ContextualRelabelingReplayBuffer,
    RemapKeyFn,
)
from rlkit.data_management.online_offline_split_replay_buffer import (
    OnlineOfflineSplitReplayBuffer,
)
from rlkit.envs.contextual import ContextualEnv
from rlkit.envs.contextual.goal_conditioned import (
    GoalDictDistributionFromMultitaskEnv,
    ContextualRewardFnFromMultitaskEnv,
    AddImageDistribution,
    PresampledPathDistribution,
    NotDonePresampledPathDistribution,
)
from rlkit.envs.contextual.latent_distributions import (
    AmortizedConditionalPriorDistribution,
    PresampledPriorDistribution,
    ConditionalPriorDistribution,
    AmortizedPriorDistribution,
    AddDecodedImageDistribution,
    AddLatentDistribution,
    AddGripperStateDistribution,
    AddConditionalLatentDistribution,
    PriorDistribution,
    PresamplePriorDistribution,
)
from rlkit.envs.images import EnvRenderer, InsertImageEnv
from rlkit.launchers.rl_exp_launcher_util import create_exploration_policy
from rlkit.samplers.data_collector.contextual_path_collector import (
    ContextualPathCollector
)
from rlkit.envs.encoder_wrappers import EncoderWrappedEnv, ConditionalEncoderWrappedEnv
from rlkit.envs.gripper_state_wrapper import GripperStateWrappedEnv
from rlkit.envs.dual_encoder_wrapper import DualEncoderWrappedEnv
from rlkit.envs.vae_wrappers import VAEWrappedEnv
from rlkit.core.eval_util import create_stats_ordered_dict
from collections import OrderedDict

from multiworld.core.image_env import ImageEnv, unormalize_image
import multiworld

from rlkit.torch.grill.common import train_vae
from rlkit.torch.gan.bigan import CVBiGAN
from rlkit.torch.vae.vq_vae import CCVAE

from roboverse.bullet.misc import quat_to_deg
from rlkit.samplers.rollout_functions import contextual_rollout

class RewardFn:
    def __init__(self,
            env,
            obs_type='latent',
            reward_type='dense',
            epsilon=1.0,
            # cnn=None,
    ):
        if obs_type == 'latent':
            self.observation_key = 'latent_observation'
            self.desired_goal_key = 'latent_desired_goal'
        elif obs_type == 'state':
            self.observation_key = 'state_observation'
            self.desired_goal_key = 'state_desired_goal'
        self.env = env
        self.reward_type = reward_type
        self.epsilon = epsilon

    def process(self, obs):
        if len(obs.shape) == 1:
            return obs.reshape(1, -1)
        return obs

    def __call__(self, states, actions, next_states, contexts):
        s = self.process(next_states[self.observation_key])
        c = self.process(contexts[self.desired_goal_key])

        if self.reward_type == 'dense':
            reward = -np.linalg.norm(s - c, axis=1)
        elif self.reward_type == 'sparse':
            success = np.linalg.norm(s - c, axis=1) < self.epsilon
            reward = success - 1
        elif self.reward_type == 'wrapped_env':
            reward = self.env.compute_reward(states, actions, next_states,
                contexts)
        else:
            raise ValueError(self.reward_type)
        return reward



def compute_hand_sparse_reward(next_obs, reward, done, info):
    return info['goal_achieved'] - 1

def resume(variant):
    data = load_local_or_remote_file(variant.get("pretrained_algorithm_path"), map_location="cuda")
    algo = data['algorithm']

    algo.num_epochs = variant['num_epochs']

    post_pretrain_hyperparams = variant["trainer_kwargs"].get("post_pretrain_hyperparams", {})
    algo.trainer.set_algorithm_weights(**post_pretrain_hyperparams)

    algo.train()

def process_args(variant):
    if variant.get("debug", False):
        variant['max_path_length'] = 5
        variant['num_presample'] = 50
        variant.get('algo_kwargs', {}).update(dict(
            batch_size=64,
            start_epoch=-100,
            num_epochs=5,
            num_eval_steps_per_epoch=16,
            num_expl_steps_per_train_loop=2,
            num_trains_per_train_loop=2,
            num_online_trains_per_train_loop=2,
            min_num_steps_before_training=2,
        ))
        variant['online_offline_split_replay_buffer_kwargs']['online_replay_buffer_kwargs']['max_size'] = int(5E2)
        variant['online_offline_split_replay_buffer_kwargs']['offline_replay_buffer_kwargs']['max_size'] = int(5E2)
        variant.get('train_vae_kwargs', {}).update(dict(
            num_epochs=1,
            train_pixelcnn_kwargs=dict(
                num_epochs=1,
                data_size=10,
                num_train_batches_per_epoch=2,
                num_test_batches_per_epoch=2,
            ),
        ))
        demo_paths = variant['path_loader_kwargs'].get('demo_paths', [])
        if len(demo_paths) > 1:
            variant['path_loader_kwargs']['demo_paths'] = [demo_paths[0]]

def iql_rig_experiment(
        max_path_length,
        qf_kwargs,
        vf_kwargs,
        trainer_kwargs,
        replay_buffer_kwargs,
        online_offline_split_replay_buffer_kwargs,
        policy_kwargs,
        algo_kwargs,
        train_vae_kwargs,
        image=False,
        online_offline_split=False,
        policy_class=TanhGaussianPolicy,
        env_id=None,
        env_class=None,
        env_kwargs=None,
        reward_kwargs=None,
        encoder_wrapper=EncoderWrappedEnv,
        observation_key='latent_observation',
        observation_keys=['latent_observation'],
        observation_key_reward_fn=None,
        desired_goal_key='latent_desired_goal',
        desired_goal_key_reward_fn=None,
        state_observation_key='state_observation',
        gripper_observation_key='gripper_state_observation',
        state_goal_key='state_desired_goal',
        image_goal_key='image_desired_goal',
        gripper_goal_key='gripper_state_desired_goal',
        reset_keys_map=None,
        gripper_observation=False,

        path_loader_class=MDPPathLoader,
        demo_replay_buffer_kwargs=None,
        path_loader_kwargs=None,
        env_demo_path='',
        env_offpolicy_data_path='',

        debug=False,
        epsilon=1.0,
        exploration_policy_kwargs=None,
        evaluation_goal_sampling_mode=None,
        exploration_goal_sampling_mode=None,
        training_goal_sampling_mode=None,

        add_env_demos=False,
        add_env_offpolicy_data=False,
        save_paths=False,
        load_demos=False,
        pretrain_policy=False,
        pretrain_rl=False,
        save_pretrained_algorithm=False,
        train_model_func=train_vae,

        # Video parameters
        save_video=True,
        expl_save_video_kwargs=None,
        eval_save_video_kwargs=None,
        renderer_kwargs=None,
        imsize=84,
        pretrained_vae_path="",
        pretrained_rl_path="",
        input_representation="",
        goal_representation="",
        presampled_goal_kwargs=None,
        presampled_goals_path="",
        ccvae_or_cbigan_exp=False,
        num_presample=5000,
        init_camera=None,
        qf_class=ConcatMlp,
        vf_class=Mlp,
        env_type=None, # For plotting
        seed=None,
        **kwargs,
    ):

    #Kwarg Definitions
    if exploration_policy_kwargs is None:
        exploration_policy_kwargs = {}
    if reset_keys_map is None:
        reset_keys_map = {}
    if demo_replay_buffer_kwargs is None:
        demo_replay_buffer_kwargs = {}
    if presampled_goal_kwargs is None:
        presampled_goal_kwargs = \
            {'eval_goals': '','expl_goals': '','training_goals': ''}
    if path_loader_kwargs is None:
        path_loader_kwargs = {}
    if not expl_save_video_kwargs:
        expl_save_video_kwargs = {}
    if not eval_save_video_kwargs:
        eval_save_video_kwargs = {}
    if not renderer_kwargs:
        renderer_kwargs = {}
    
    #Enviorment Wrapping
    renderer = EnvRenderer(init_camera=init_camera, **renderer_kwargs)
    def contextual_env_distrib_and_reward(
            env_id, env_class, env_kwargs, encoder_wrapper, goal_sampling_mode, presampled_goals_path, num_presample
    ):
        state_env = get_gym_env(env_id, env_class=env_class, env_kwargs=env_kwargs)
        renderer = EnvRenderer(init_camera=init_camera, **renderer_kwargs)
        img_env = InsertImageEnv(state_env, renderer=renderer)

        encoded_env = encoder_wrapper(
            img_env,
            model,
            step_keys_map=dict(image_observation="latent_observation"),
            reset_keys_map=reset_keys_map,
        )
        if gripper_observation:
            encoded_env = GripperStateWrappedEnv(
                encoded_env, 
                state_observation_key,
                step_keys_map=dict(gripper_state_observation="gripper_state_observation")
            )
        if goal_sampling_mode == "vae_prior":
            latent_goal_distribution = PriorDistribution(
                model.representation_size,
                desired_goal_key_reward_fn if desired_goal_key_reward_fn else desired_goal_key,
            )
            diagnostics = StateImageGoalDiagnosticsFn({}, )

        elif goal_sampling_mode == "amortized_vae_prior":
            latent_goal_distribution = AmortizedPriorDistribution(
                model,
                desired_goal_key_reward_fn if desired_goal_key_reward_fn else desired_goal_key,
                num_presample=num_presample,
            )
            diagnostics = StateImageGoalDiagnosticsFn({}, )

        elif goal_sampling_mode == 'conditional_vae_prior':
            latent_goal_distribution = ConditionalPriorDistribution(
                model,
                desired_goal_key_reward_fn if desired_goal_key_reward_fn else desired_goal_key
            )
            if image:
                latent_goal_distribution = AddDecodedImageDistribution(
                    latent_goal_distribution,
                    desired_goal_key_reward_fn if desired_goal_key_reward_fn else desired_goal_key,
                    image_goal_key,
                    model,
                )
            diagnostics = StateImageGoalDiagnosticsFn({}, )
        elif goal_sampling_mode == "amortized_conditional_vae_prior":
            latent_goal_distribution = AmortizedConditionalPriorDistribution(
                model,
                desired_goal_key_reward_fn if desired_goal_key_reward_fn else desired_goal_key,
                num_presample=num_presample,
            )
            diagnostics = StateImageGoalDiagnosticsFn({}, )
        elif goal_sampling_mode == "presampled_images":
            diagnostics = state_env.get_contextual_diagnostics
            image_goal_distribution = PresampledPathDistribution(
                presampled_goals_path,
                model.representation_size,
            )

            #Representation Check
            if ccvae_or_cbigan_exp:
                add_distrib = AddConditionalLatentDistribution
            else:
                add_distrib = AddLatentDistribution

            #AddLatentDistribution
            latent_goal_distribution = add_distrib(
                image_goal_distribution,
                image_goal_key,
                desired_goal_key_reward_fn if desired_goal_key_reward_fn else desired_goal_key,
                model,
            )
        elif goal_sampling_mode == "not_done_presampled_images":
            diagnostics = state_env.get_contextual_diagnostics
            image_goal_distribution = NotDonePresampledPathDistribution(
                presampled_goals_path,
                model.representation_size,
                encoded_env,
            )

            #Representation Check
            if ccvae_or_cbigan_exp:
                add_distrib = AddConditionalLatentDistribution
            else:
                add_distrib = AddLatentDistribution

            #AddLatentDistribution
            latent_goal_distribution = add_distrib(
                image_goal_distribution,
                image_goal_key,
                desired_goal_key_reward_fn if desired_goal_key_reward_fn else desired_goal_key,
                model,
            )
        elif goal_sampling_mode == "presample_latents":
            diagnostics = StateImageGoalDiagnosticsFn({}, )
            #diagnostics = state_env.get_contextual_diagnostics
            latent_goal_distribution = PresamplePriorDistribution(
                model,
                desired_goal_key_reward_fn if desired_goal_key_reward_fn else desired_goal_key,
                state_env,
                num_presample=num_presample,
            )
            if image:
                latent_goal_distribution = AddDecodedImageDistribution(
                    latent_goal_distribution,
                    desired_goal_key_reward_fn if desired_goal_key_reward_fn else desired_goal_key,
                    image_goal_key,
                    model,
                )
        elif goal_sampling_mode == "presampled_latents":
            diagnostics = state_env.get_contextual_diagnostics
            latent_goal_distribution = PresampledPriorDistribution(
                presampled_goals_path,
                desired_goal_key_reward_fn if desired_goal_key_reward_fn else desired_goal_key,
            )
        elif goal_sampling_mode == "reset_of_env":
            state_goal_env = get_gym_env(env_id, env_class=env_class, env_kwargs=env_kwargs)
            state_goal_distribution = GoalDictDistributionFromMultitaskEnv(
                state_goal_env,
                desired_goal_keys=[state_goal_key],
            )
            image_goal_distribution = AddImageDistribution(
                env=state_env,
                base_distribution=state_goal_distribution,
                image_goal_key=image_goal_key,
                renderer=renderer,
            )
            latent_goal_distribution = AddLatentDistribution(
                image_goal_distribution,
                image_goal_key,
                desired_goal_key_reward_fn if desired_goal_key_reward_fn else desired_goal_key,
                model,
            )
            diagnostics = state_goal_env.get_contextual_diagnostics
        else:
            error
        
        if gripper_observation:
            latent_goal_distribution = AddGripperStateDistribution(
                latent_goal_distribution,
                state_goal_key,
                gripper_goal_key,
            )

        reward_fn = RewardFn(
            state_env,
            **reward_kwargs
        )
        env = ContextualEnv(
            encoded_env,
            context_distribution=latent_goal_distribution,
            reward_fn=reward_fn,
            observation_key=observation_key,
            contextual_diagnostics_fns=[diagnostics],
        )
        return env, latent_goal_distribution, reward_fn

    if pretrained_vae_path:
        model = load_local_or_remote_file(pretrained_vae_path)
        path_loader_kwargs['model_path'] = pretrained_vae_path
    else:
        model = train_model_func(train_vae_kwargs)
        path_loader_kwargs['model'] = model

    #Representation Check
    if isinstance(model, CCVAE) or isinstance(model, CVBiGAN):
        ccvae_or_cbigan_exp = True
        path_loader_kwargs['condition_encoding'] = True
        encoder_wrapper = ConditionalEncoderWrappedEnv
        exploration_goal_sampling_mode = "conditional_vae_prior"

    #Environment Definitions
    expl_env_kwargs = env_kwargs.copy()
    expl_env_kwargs['expl'] = True

    eval_env_kwargs = env_kwargs.copy()
    eval_env_kwargs['expl'] = False

    expl_env, expl_context_distrib, expl_reward = contextual_env_distrib_and_reward(
        env_id, env_class, expl_env_kwargs, encoder_wrapper, exploration_goal_sampling_mode,
        presampled_goal_kwargs['expl_goals'], num_presample
    )
    training_env, training_context_distrib, training_reward = contextual_env_distrib_and_reward(
        env_id, env_class, expl_env_kwargs, encoder_wrapper, training_goal_sampling_mode,
        presampled_goal_kwargs['training_goals'], num_presample
    )
    eval_env, eval_context_distrib, eval_reward = contextual_env_distrib_and_reward(
        env_id, env_class, eval_env_kwargs, encoder_wrapper, evaluation_goal_sampling_mode,
        presampled_goal_kwargs['eval_goals'], num_presample
    )
    path_loader_kwargs['env'] = eval_env

    #IQL Code
    if add_env_demos:
        path_loader_kwargs["demo_paths"].append(env_demo_path)
    if add_env_offpolicy_data:
        path_loader_kwargs["demo_paths"].append(env_offpolicy_data_path)

    #Key Setting
    context_key = desired_goal_key
    obs_dim = (
            expl_env.observation_space.spaces[observation_key].low.size
            + expl_env.observation_space.spaces[context_key].low.size
    )
    if gripper_observation:
        obs_dim += 6*2
    action_dim = expl_env.action_space.low.size

    state_rewards = reward_kwargs.get('reward_type', 'dense') == 'wrapped_env'
    
    mapper_dict = {context_key: observation_key}
    obs_keys = [observation_key]
    cont_keys = [context_key]

    if desired_goal_key_reward_fn:
        mapper_dict[desired_goal_key_reward_fn] = observation_key_reward_fn
        obs_keys.append(observation_key_reward_fn)
        cont_keys.append(desired_goal_key_reward_fn)

    obs_keys.append(state_observation_key)
    if state_rewards:
        mapper_dict[state_goal_key] = state_observation_key
        cont_keys.append(state_goal_key)
    else:
        obs_keys.extend(list(reset_keys_map.values()))
    
    if gripper_observation:
        mapper_dict[gripper_goal_key] = gripper_observation_key
        obs_keys.append(gripper_observation_key)
        cont_keys.append(gripper_goal_key)

    mapper = RemapKeyFn(mapper_dict)
    

    #Replay Buffer
    def concat_context_to_obs(batch, replay_buffer, obs_dict, next_obs_dict, new_contexts):
        obs = batch['observations']
        if type(obs) is tuple:
            obs = np.concatenate(obs, axis=1)
        next_obs = batch['next_observations']
        if type(next_obs) is tuple:
            next_obs = np.concatenate(next_obs, axis=1)
        if len(new_contexts.keys()) > 1:
            context = np.concatenate(tuple(new_contexts.values()), axis=1)
        else:
            context = batch[context_key]
        batch['observations'] = np.concatenate([obs, context], axis=1)
        batch['next_observations'] = np.concatenate([next_obs, context], axis=1)
        return batch

    if online_offline_split:
        online_replay_buffer = ContextualRelabelingReplayBuffer(
            env=eval_env,
            context_keys=cont_keys,
            observation_keys_to_save=obs_keys,
            observation_key=observation_key if observation_keys is None else None,
            observation_key_reward_fn=observation_key_reward_fn,
            observation_keys=observation_keys,
            context_distribution=training_context_distrib,#expl_context_distrib,
            sample_context_from_obs_dict_fn=mapper,
            reward_fn=eval_reward,
            post_process_batch_fn=concat_context_to_obs,
            **online_offline_split_replay_buffer_kwargs['online_replay_buffer_kwargs']
        )
        offline_replay_buffer = ContextualRelabelingReplayBuffer(
            env=eval_env,
            context_keys=cont_keys,
            observation_keys_to_save=obs_keys,
            observation_key=observation_key if observation_keys is None else None,
            observation_key_reward_fn=observation_key_reward_fn,
            observation_keys=observation_keys,
            context_distribution=training_context_distrib,#expl_context_distrib,
            sample_context_from_obs_dict_fn=mapper,
            reward_fn=eval_reward,
            post_process_batch_fn=concat_context_to_obs,
            **online_offline_split_replay_buffer_kwargs['offline_replay_buffer_kwargs']
        )
        replay_buffer = OnlineOfflineSplitReplayBuffer(
            offline_replay_buffer,
            online_replay_buffer,
            **online_offline_split_replay_buffer_kwargs
        )
    else:
        replay_buffer = ContextualRelabelingReplayBuffer(
            env=eval_env,
            context_keys=cont_keys,
            observation_keys_to_save=obs_keys,
            observation_key=observation_key if observation_keys is None else None,
            observation_key_reward_fn=observation_key_reward_fn,
            observation_keys=observation_keys,
            context_distribution=training_context_distrib,#expl_context_distrib,
            sample_context_from_obs_dict_fn=mapper,
            reward_fn=eval_reward,
            post_process_batch_fn=concat_context_to_obs,
            **replay_buffer_kwargs
        )

    if pretrained_rl_path:
        rl_model_dict = load_local_or_remote_file(pretrained_rl_path)
        qf1 = rl_model_dict['trainer/qf1']
        qf2 = rl_model_dict['trainer/qf2']
        target_qf1 = rl_model_dict['trainer/target_qf1']
        target_qf2 = rl_model_dict['trainer/target_qf2']
        vf = rl_model_dict['trainer/vf']
        policy = rl_model_dict['trainer/policy']
        if 'std' in policy_kwargs and policy_kwargs['std'] is not None:
            policy.std = policy_kwargs['std']
            policy.log_std = np.log(policy.std)
    else:
        #Neural Network Architecture
        def create_qf():
            if qf_class is ConcatMlp:
                qf_kwargs["input_size"] = obs_dim + action_dim
            if qf_class is ConcatCNN:
                qf_kwargs["added_fc_input_size"] = action_dim
            return qf_class(
                output_size=1,
                **qf_kwargs
            )
        qf1 = create_qf()
        qf2 = create_qf()
        target_qf1 = create_qf()
        target_qf2 = create_qf()

        def create_vf():
            if vf_class is Mlp:
                vf_kwargs["input_size"] = obs_dim
            return vf_class(
                output_size=1,
                **vf_kwargs
            )
        vf = create_vf()

        policy = policy_class(
            obs_dim=obs_dim,
            action_dim=action_dim,
            **policy_kwargs,
        )

    #Path Collectors
    path_collector_observation_keys = [observation_key, ] if observation_keys is None else observation_keys
    path_collector_context_keys_for_policy = [context_key, ] if not gripper_observation else [context_key, gripper_goal_key]
    def obs_processor(o):
        combined_obs = []
        for k in path_collector_observation_keys:
            if k == gripper_observation_key:
                combined_obs.append(np.concatenate((o['state_observation'][:3], quat_to_deg(o['state_observation'][3:7])/360.0), axis=0))
            else:
                combined_obs.append(o[k])
        for k in path_collector_context_keys_for_policy:
            if k == gripper_goal_key:
                combined_obs.append(np.concatenate((o['state_desired_goal'][:3], quat_to_deg(o['state_desired_goal'][3:7])/360.0), axis=0))
            else:
                combined_obs.append(o[k])
        return np.concatenate(combined_obs, axis=0)

    def gripper_state_contextual_rollout(
        env,
        agent,
        **kwargs,
    ):
        paths = contextual_rollout(
            env,
            agent,
            **kwargs
        )
        
        for i in range(paths['observations'].shape[0]):
            d = paths['observations'][i]
            d['gripper_state_observation'] = np.concatenate((d['state_observation'][:3], quat_to_deg(d['state_observation'][3:7])/360.0), axis=0)
            d['gripper_state_desired_goal'] = np.concatenate((d['state_desired_goal'][:3], quat_to_deg(d['state_desired_goal'][3:7])/360.0), axis=0)
        
        for i in range(paths['next_observations'].shape[0]):
            d = paths['next_observations'][i]
            d['gripper_state_observation'] = np.concatenate((d['state_observation'][:3], quat_to_deg(d['state_observation'][3:7])/360.0), axis=0)
            d['gripper_state_desired_goal'] = np.concatenate((d['state_desired_goal'][:3], quat_to_deg(d['state_desired_goal'][3:7])/360.0), axis=0)
        
        return paths

    eval_path_collector = ContextualPathCollector(
        eval_env,
        MakeDeterministic(policy),
        observation_keys=path_collector_observation_keys,
        context_keys_for_policy=path_collector_context_keys_for_policy,
        obs_processor=obs_processor,
        rollout=gripper_state_contextual_rollout if gripper_observation else contextual_rollout,
    )
    exploration_policy = create_exploration_policy(
        expl_env, policy, **exploration_policy_kwargs)
    expl_path_collector = ContextualPathCollector(
        expl_env,
        exploration_policy,
        observation_keys=path_collector_observation_keys,
        context_keys_for_policy=path_collector_context_keys_for_policy,
        obs_processor=obs_processor,
        rollout=gripper_state_contextual_rollout if gripper_observation else contextual_rollout,
    )

    if trainer_kwargs['use_online_beta']:
        assert trainer_kwargs['use_anneal_beta'] == False
        if algo_kwargs['start_epoch'] == 0:
            trainer_kwargs['beta'] = trainer_kwargs['beta_online']

    #Algorithm
    trainer = IQLTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        vf=vf,
        **trainer_kwargs
    )

    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        max_path_length=max_path_length,
        **algo_kwargs
    )

    if trainer_kwargs['use_online_beta']:
        assert trainer_kwargs['use_anneal_beta'] == False
        def switch_beta(self, epoch):
            if epoch == -1:
                self.trainer.beta = trainer_kwargs['beta_online']
        algorithm.post_epoch_funcs.append(switch_beta)
    elif trainer_kwargs['use_anneal_beta']:
        def switch_beta(self, epoch):
            print(self.trainer.beta)
            if (epoch != algo_kwargs['start_epoch'] 
                and (epoch - algo_kwargs['start_epoch']) % trainer_kwargs['anneal_beta_every'] == 0 
                and self.trainer.beta * trainer_kwargs['anneal_beta_by'] >= trainer_kwargs['anneal_beta_stop_at']):
                 self.trainer.beta *= trainer_kwargs['anneal_beta_by']
        algorithm.post_epoch_funcs.append(switch_beta)

    algorithm.to(ptu.device)

    #Video Saving
    if save_video:
        expl_video_func = RIGVideoSaveFunction(
            model,
            expl_path_collector,
            "train",
            decode_goal_image_key="image_decoded_goal",
            reconstruction_key="image_reconstruction",
            rows=2,
            columns=5,
            unnormalize=True,
            imsize=imsize,
            image_format=renderer.output_image_format,
            **expl_save_video_kwargs
        )
        algorithm.post_train_funcs.append(expl_video_func)

        eval_video_func = RIGVideoSaveFunction(
            model,
            eval_path_collector,
            "eval",
            goal_image_key=image_goal_key,
            decode_goal_image_key="image_decoded_goal",
            reconstruction_key="image_reconstruction",
            num_imgs=4,
            rows=2,
            columns=5,
            unnormalize=True,
            imsize=imsize,
            image_format=renderer.output_image_format,
            **eval_save_video_kwargs
        )
        algorithm.post_train_funcs.append(eval_video_func)

    #IQL CODE
    if save_paths:
        algorithm.post_train_funcs.append(save_paths)

    if online_offline_split:
        replay_buffer.set_online_mode(False)
    if load_demos:
        demo_train_buffer = None 
        demo_test_buffer = None
        path_loader = path_loader_class(trainer,
            replay_buffer=replay_buffer,
            demo_train_buffer=demo_train_buffer,
            demo_test_buffer=demo_test_buffer,
            reward_fn=eval_reward,
            **path_loader_kwargs
        )
        path_loader.load_demos()

    if save_pretrained_algorithm:
        p_path = osp.join(logger.get_snapshot_dir(), 'pretrain_algorithm.p')
        pt_path = osp.join(logger.get_snapshot_dir(), 'pretrain_algorithm.pt')
        data = algorithm._get_snapshot()
        data['algorithm'] = algorithm
        torch.save(data, open(pt_path, "wb"))
        torch.save(data, open(p_path, "wb"))

    if online_offline_split:
        replay_buffer.set_online_mode(True)
    
    algorithm.train()