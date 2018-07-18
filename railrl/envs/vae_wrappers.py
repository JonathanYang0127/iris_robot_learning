import pickle
import cv2
import numpy as np
from gym import Env
from gym.spaces import Box, Dict
import railrl.torch.pytorch_util as ptu
from multiworld.envs.env_util import get_stat_in_paths, create_stats_ordered_dict
from multiworld.envs.mujoco.sawyer_reach_torque.generate_goal_data_set import generate_goal_data_set
from railrl.envs.wrappers import ProxyEnv
from railrl.misc.asset_loader import sync_down


def load_vae(vae_file):
    if vae_file[0] == "/":
        local_path = vae_file
    else:
        local_path = sync_down(vae_file)
    vae = pickle.load(open(local_path, "rb"))
    # vae = torch.load(local_path, map_location=lambda storage, loc: storage)
    print("loaded", local_path)
    return vae


class VAEWrappedEnv(ProxyEnv, Env):
    """This class wraps an image-based environment with a VAE.
    Assumes you get flattened (channels,84,84) observations from wrapped_env.
    This class adheres to the "Silent Multitask Env" semantics: on reset,
    it resamples a goal.
    """
    def __init__(
        self,
        wrapped_env,
        vae,
        observation_key='latent_observation',
        use_vae_goals=True,
        sample_from_true_prior=False,
        decode_goals=False,
        render_goals=False,
        render_rollouts=False,
        reward_params=None,
        mode="train",
        imsize=84,
        use_goal_caching=False,
        cached_goal_generation_function=generate_goal_data_set,
        num_cached_goals=100,
        cached_goal_keys=None,
        goal_sizes=None,
        obs_to_goal_fctns=None,
        observation_keys=None,
        use_cached_dataset=False,
        num_goals_presampled=0,
    ):
        self.quick_init(locals())
        if reward_params is None:
            reward_params = dict()
        super().__init__(wrapped_env)
        if type(vae) is str:
            self.vae = load_vae(vae)
        else:
            self.vae = vae
        if ptu.gpu_enabled():
            vae.cuda()
        self.representation_size = self.vae.representation_size
        self.input_channels = self.vae.input_channels
        self._use_vae_goals = use_vae_goals
        self.sample_from_true_prior = sample_from_true_prior
        self.decode_goals = decode_goals
        self.render_goals = render_goals
        self.render_rollouts = render_rollouts
        self.default_kwargs=dict(
            decode_goals=decode_goals,
            render_goals=render_goals,
            render_rollouts=render_rollouts,
        )
        self.imsize = imsize
        self.num_goals_presampled = num_goals_presampled

        self.reward_params = reward_params
        self.reward_type = self.reward_params.get("type", 'latent_distance')
        self.norm_order = self.reward_params.get("norm_order", 1)
        self.epsilon = self.reward_params.get("epsilon", 20)
        self.reward_min_variance = self.reward_params.get("min_variance", 0)
        latent_space = Box(
            -10 * np.ones(self.representation_size),
            10 * np.ones(self.representation_size),
        )
        spaces = self.wrapped_env.observation_space.spaces
        spaces['observation'] = latent_space
        spaces['desired_goal'] = latent_space
        spaces['achieved_goal'] = latent_space
        spaces['latent_observation'] = latent_space
        spaces['latent_desired_goal'] = latent_space
        spaces['latent_achieved_goal'] = latent_space
        self.observation_space = Dict(spaces)
        self._latent_goal = np.random.uniform(0, 1, self.representation_size)
        self._vw_goal_img = None
        self._vw_goal_img_decoded = None
        self.use_goal_caching = use_goal_caching
        self.num_cached_goals = num_cached_goals
        if self.use_goal_caching:
            # hardcoded for torque control for now
            self.wrapped_env._img_goal=np.random.uniform(0, 1, self.image_length)
            cached_goal_keys = ['latent_desired_goal', 'image_desired_goal', 'state_desired_goal', 'joint_desired_goal']
            goal_sizes = [self.representation_size, (self.imsize ** 2) * 3, 3, 7]
            obs_to_goal_fctns = [lambda x: x, lambda x: x, lambda x: x[-3:], lambda x: x[:7]]
            observation_keys = ['latent_observation', 'image_observation', 'state_observation', 'state_observation']
            goal_generation_dict = dict()
            for goal_key, goal_size, obs_to_goal_fctn, obs_key in zip(cached_goal_keys, goal_sizes, obs_to_goal_fctns,
                                                                      observation_keys):
                goal_generation_dict[goal_key] = [goal_size, obs_to_goal_fctn, obs_key]
            self.goals = cached_goal_generation_function(self, goal_generation_dict=goal_generation_dict, num_goals=num_cached_goals, use_cached_dataset=use_cached_dataset)
            self.goals['desired_goal'] = self.goals['latent_desired_goal']
            self._wrapped_env._wrapped_env.goals = self.goals
            self._wrapped_env._wrapped_env.use_goal_caching = True

        self.mode(mode)
        self._presampled_goals = None

        self._mode_map = {}

    @property
    def use_vae_goals(self):
        return self._use_vae_goals and not self.reward_type.startswith('state')

    def mode(self, name):
        if name == "train":
            self._use_vae_goals = True
            self.decode_goals = self.default_kwargs['decode_goals']
            self.render_goals = self.default_kwargs['render_goals']
            self.render_rollouts = self.default_kwargs['render_rollouts']
        elif name == "train_env_goals":
            self._use_vae_goals = False
            self.decode_goals = self.default_kwargs['decode_goals']
            self.render_goals = self.default_kwargs['render_goals']
            self.render_rollouts = self.default_kwargs['render_rollouts']
        elif name == "test":
            self._use_vae_goals = False
            self.decode_goals = self.default_kwargs['decode_goals']
            self.render_goals = self.default_kwargs['render_goals']
            self.render_rollouts = self.default_kwargs['render_rollouts']
        elif name == "video_vae":
            self._use_vae_goals = True
            self.decode_goals = True
            self.render_goals = False
            self.render_rollouts = False
        elif name == "video_env":
            self._use_vae_goals = False
            self.decode_goals = False
            self.render_goals = False
            self.render_rollouts = False
            self.render_decoded = False
        else:
            raise ValueError("Invalid mode: {}".format(name))
        self.cur_mode = name

    def add_mode(self, env_type, mode):
        assert env_type in ['train',
                        'eval',
                        'video_vae',
                        'video_env',
                        'relabeling']
        assert mode in ['train',
                        'train_env_goals',
                        'test',
                        'video_vae',
                        'video_env']
        assert env_type not in self._mode_map
        self._mode_map[env_type] = mode

    def train(self):
        self.mode(self._mode_map['train'])

    def eval(self):
        self.mode(self._mode_map['eval'])

    @property
    def goal_dim(self):
        return self.representation_size

    def step(self, action):
        obs, reward, done, info = self.wrapped_env.step(action)
        new_obs = self._update_obs(obs)
        self._update_info(info, obs)
        reward = self.compute_reward(
            action,
            {'latent_achieved_goal': new_obs['latent_achieved_goal'],
             'latent_desired_goal': new_obs['latent_desired_goal']}
        )
        if self.render_rollouts:
            img = obs['image_observation'].reshape(
                self.input_channels,
                self.imsize,
                self.imsize,
            ).transpose()
            cv2.imshow('env', img)
            cv2.waitKey(1)
            reconstruction = self._reconstruct_img(obs['image_observation'])
            cv2.imshow('env_reconstruction', reconstruction)
            cv2.waitKey(1)
        if self.render_goals and not self.use_vae_goals:
            goal = self._vw_goal_img.reshape(
                self.input_channels,
                self.imsize,
                self.imsize,
            ).transpose()
            cv2.imshow('goal', goal)
            cv2.waitKey(1)

        if self.render_goals:
            cv2.imshow('decoded', self._vw_goal_img_decoded)
            cv2.waitKey(1)
        return new_obs, reward, done, info

    def _reconstruct_img(self, flat_img):
        zs = self.vae.encode(ptu.np_to_var(flat_img[None]))[0]
        imgs = ptu.get_numpy(self.vae.decode(zs))
        imgs = imgs.reshape(
            1, self.input_channels, 84, 84
        ).transpose([0, 3, 2, 1])
        return imgs[0]

    def _update_info(self, info, obs):
        latent_obs, logvar = self.vae.encode(ptu.np_to_var(obs['image_observation']))
        latent_obs, logvar = ptu.get_numpy(latent_obs), ptu.get_numpy(logvar)
        assert (latent_obs == obs['latent_observation']).all()
        latent_goal = self._latent_goal
        dist = latent_goal - latent_obs
        var = np.exp(logvar.flatten())
        var = np.maximum(var, self.reward_min_variance)
        err = dist * dist / 2 / var
        mdist = np.sum(err)  # mahalanobis distance
        info["vae_mdist"] = mdist
        info["vae_success"] = 1 if mdist < self.epsilon else 0
        info["vae_dist"] = np.linalg.norm(dist, ord=self.norm_order)
        info["vae_dist_l1"] = np.linalg.norm(dist, ord=1)
        info["vae_dist_l2"] = np.linalg.norm(dist, ord=2)

    def reset(self):
        obs = self.wrapped_env.reset()
        if self.use_vae_goals:
            latent_goals = self._sample_vae_prior(1)
            if self.decode_goals:
                goal_img = self._decode(latent_goals)[0].transpose().flatten()
            else:
                goal_img = None
            obs['image_desired_goal'] = goal_img
            obs['state_desired_goal'] = None
            self._latent_goal = latent_goals[0]
            self._vw_goal_img = goal_img
            self._vw_goal_img_decoded = goal_img
        else:
            if self.use_goal_caching:
                idx = np.random.randint(0, self.num_cached_goals)
                self._latent_goal = self.goals['latent_desired_goal'][idx]
                self._wrapped_env._wrapped_env._state_goal = self.goals['state_desired_goal'][idx]
                self._wrapped_env._wrapped_env._goal_angles = self.goals['joint_desired_goal'][idx]
                self._wrapped_env._img_goal = self.goals['image_desired_goal'][idx]
                for key in self.goals.keys():
                    obs[key] = self.goals[key][idx]
            else:
                self._latent_goal = self._encode_one(obs['image_desired_goal'])
            if self.decode_goals:
                self._vw_goal_img_decoded = self._decode(self._latent_goal[None])[0]
        self._vw_goal_img = obs['image_desired_goal']
        obs['desired_goal'] = self._latent_goal
        obs['latent_desired_goal'] = self._latent_goal
        return self._update_obs(obs)

    def enable_render(self):
        self._use_vae_goals = False
        self.decode_goals = True
        self.render_goals = True
        self.render_rollouts = True

    def disable_render(self):
        self.decode_goals = False
        self.render_goals = False
        self.render_rollouts = False

    def _update_obs(self, obs):
        latent_obs = self._encode_one(obs['image_observation'])
        obs['latent_observation'] = latent_obs
        obs['latent_desired_goal'] = self._latent_goal
        obs['latent_achieved_goal'] = latent_obs
        obs['observation'] = latent_obs
        obs['desired_goal'] = self._latent_goal
        obs['achieved_goal'] = latent_obs
        obs['image_desired_goal'] = self._vw_goal_img
        return obs

    def _sample_vae_prior(self, batch_size):
        if self.sample_from_true_prior:
            mu, sigma = 0, 1  # sample from prior
        else:
            mu, sigma = self.vae.dist_mu, self.vae.dist_std
        n = np.random.randn(batch_size, self.representation_size)
        return sigma * n + mu

    def _decode(self, latents):
        batch_size = latents.shape[0]
        imgs = ptu.get_numpy(self.vae.decode(ptu.np_to_var(latents)))
        # TODO: why do we need this tranpose? Can we eliminate it?
        imgs = imgs.reshape(
            batch_size, self.input_channels, 84, 84
        ).transpose([0, 3, 2, 1])
        return imgs

    def _encode_one(self, img):
        return self._encode(img[None])[0]

    def _encode(self, imgs):
        return ptu.get_numpy(self.vae.encode(ptu.np_to_var(imgs))[0])

    """
    Multitask functions
    """
    def get_goal(self):
        goal = self.wrapped_env.get_goal()
        goal['desired_goal'] = self._latent_goal
        goal['latent_desired_goal'] = self._latent_goal
        return goal

    def sample_goals(self, batch_size, force_resample=False):
        if self.use_goal_caching and not self.use_vae_goals:
            idxs = np.random.randint(0, self.num_cached_goals, batch_size)
            goals = dict()
            for key in self.goals.keys():
                goals[key] = self.goals[key][idxs]
        elif (
            not force_resample
            and batch_size > 1
            and self.num_goals_presampled > 0
            and not self.use_vae_goals
        ):
            if (
                self._presampled_goals is None
                    or self.num_goals_presampled < batch_size
            ):
                self.num_goals_presampled = max(
                    self.num_goals_presampled,
                    batch_size,
                )
                self._presampled_goals = self.sample_goals(
                    self.num_goals_presampled,
                    force_resample=True,
                )

            idx = np.random.randint(0, self.num_goals_presampled, batch_size)
            sampled_goals = {
                k: v[idx] for k, v in self._presampled_goals.items()
            }
            return sampled_goals
        if self.use_vae_goals:
            goals = {}
            latent_goals = self._sample_vae_prior(batch_size)
            if self.decode_goals:
                raise NotImplementedError(
                    'I think we should change _decode to not transpose...'
                )
                goal_imgs = self._decode(latent_goals).transpose().flatten()
            else:
                goal_imgs = None
            goals['image_desired_goal'] = goal_imgs
            goals['state_desired_goal'] = None
        else:
            goals = self.wrapped_env.sample_goals(batch_size)
            latent_goals = self._encode(goals['image_desired_goal'])
        goals['desired_goal'] = latent_goals
        goals['latent_desired_goal'] = latent_goals
        return goals

    def get_diagnostics(self, paths, **kwargs):
        statistics = self.wrapped_env.get_diagnostics(paths, **kwargs)
        for stat_name_in_paths in ["vae_mdist", "vae_success", "vae_dist"]:
            stats = get_stat_in_paths(paths, 'env_infos', stat_name_in_paths)
            statistics.update(create_stats_ordered_dict(
                stat_name_in_paths,
                stats,
                always_show_all_stats=True,
            ))
            final_stats = [s[-1] for s in stats]
            statistics.update(create_stats_ordered_dict(
                "Final " + stat_name_in_paths,
                final_stats,
                always_show_all_stats=True,
            ))
        return statistics

    def compute_reward(self, action, obs):
        actions = action[None]
        next_obs = {
            k: v[None] for k, v in obs.items()
        }
        return self.compute_rewards(actions, next_obs)[0]

    def compute_rewards(self, actions, obs):
        # TODO: implement log_prob/mdist
        if self.reward_type == 'latent_distance':
            achieved_goals = obs['latent_achieved_goal']
            desired_goals = obs['latent_desired_goal']
            dist = np.linalg.norm(desired_goals - achieved_goals, ord=self.norm_order, axis=1)
            return -dist
        elif self.reward_type == 'vectorized_latent_distance':
            achieved_goals = obs['latent_achieved_goal']
            desired_goals = obs['latent_desired_goal']
            return -np.abs(desired_goals - achieved_goals)
        elif self.reward_type == 'latent_sparse':
            achieved_goals = obs['latent_achieved_goal']
            desired_goals = obs['latent_desired_goal']
            dist = np.linalg.norm(desired_goals - achieved_goals, ord=self.norm_order, axis=1)
            reward = 0 if dist < self.epsilon else -1
            return reward
        elif self.reward_type == 'state_distance':
            achieved_goals = obs['state_achieved_goal']
            desired_goals = obs['state_desired_goal']
            return - np.linalg.norm(desired_goals - achieved_goals, ord=self.norm_order, axis=1)
        elif self.reward_type == 'wrapped_env':
            return self.wrapped_env.compute_rewards(actions, obs)
        else:
            raise NotImplementedError

def temporary_mode(env, mode, func, args=None, kwargs=None):
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}
    cur_type = env.cur_mode
    env.mode(env._mode_map[mode])
    func(*args, **kwargs)
    env.mode(cur_type)
