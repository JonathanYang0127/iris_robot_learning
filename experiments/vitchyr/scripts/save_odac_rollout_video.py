import argparse
import copy
from collections import OrderedDict
from functools import partial
from pathlib import Path

import numpy as np
import torch
from gym.envs.robotics import FetchEnv
from matplotlib.ticker import ScalarFormatter

import rlkit.samplers.rollout_functions as rf
import rlkit.torch.pytorch_util as ptu
# from multiworld.envs.mujoco.classic_mujoco.hopper import \
# HopperFullPositionGoalEnv
from multiworld.envs.mujoco.cameras import create_camera_init
from multiworld.envs.mujoco.classic_mujoco.ant import (
    AntXYGoalEnv,
    AntFullPositionGoalEnv,
)
# from multiworld.envs.mujoco.classic_mujoco.hopper import \
# HopperFullPositionGoalEnv
from rlkit.envs.contextual.contextual_env import (
    batchify, ContextualEnv,
    delete_info,
)
from rlkit.envs.contextual.goal_conditioned import AddImageDistribution
from rlkit.envs.images.env_renderer import GymEnvRenderer
# from multiworld.envs.mujoco.classic_mujoco.hopper import \
# HopperFullPositionGoalEnv
from rlkit.launchers.experiments.vitchyr.probabilistic_goal_reaching.visualize import (
    DynamicNumberEnvRenderer,
    InsertDebugImagesEnv,
)
from rlkit.visualization.video import dump_video


def compute_discount_factor(
        bootstrap_value, unscaled_reward, reward_scale,
        prior_discount=0.99,
):
    discount = torch.sigmoid(
        bootstrap_value
        - unscaled_reward * reward_scale
        + np.log(prior_discount / (1 - prior_discount))
    ).detach()
    # print('discount, v, r', discount,  bootstrap_value, unscaled_reward * reward_scale)
    discount = torch.clamp(discount, max=prior_discount)
    return discount


def save_video(snapshot_path=None, save_dir=None, render=False, horizon=100, num_imgs=5):
    prefix = ''
    # snapshot_path = '/home/vitchyr/mnt2/log2/20-10-17-sawyer-door--exp1-door-sweep/20-10-17-sawyer-door--exp1-door-sweep_2020_10_17_08_20_05_id002--s194662/params.pt'
    # snapshot_path = '/home/vitchyr/mnt2/log2/20-10-17-sawyer-door--exp1-door-sweep/20-10-17-sawyer-door--exp1-door-sweep_2020_10_17_08_20_05_id002--s194662/itr_25.pt'
    # prefix='itr25_'
    # base_dir = '/home/vitchyr/mnt2/log2/paper-results/odac/sawyer-door'
    # alpha = 0.0047  # for sawyer door
    # reward_scale = 0.01  # for sawyer door

    # snapshot_path = '/home/vitchyr/mnt2/log2/20-10-17-sawyer-pnp--exp1-pnp-sweep/20-10-17-sawyer-pnp--exp1-pnp-sweep_2020_10_17_08_20_05_id002--s570237/params.pt'
    # base_dir = '/home/vitchyr/mnt2/log2/paper-results/odac/sawyer-pnp'
    # alpha = 0.0085  # for sawyer pnp
    # reward_scale = 0.043  # for sawyer p

    # base_dir = '/home/vitchyr/mnt2/log2/paper-results/odac/fetch-slide'
    # alpha = 934e-6  # for fetch slide
    # reward_scale = 0.0058  # for fetch slide

    snapshot_path = '/home/vitchyr/mnt2/log2/20-10-15-add--exp5-more-seeds-good-settings/20-10-15-add--exp5-more-seeds-good-settings_2020_10_15_21_40_24_id000--s620380/params.pt'
    base_dir = '/home/vitchyr/mnt2/log2/paper-results/odac/ant/dev'
    alpha = 0.0035  # for ant
    reward_scale = 0.07  # for ant
    # prefix = 'policy_only_'

    # snapshot_path = '/home/vitchyr/mnt2/log2/20-10-15-fetch-pnp--exp1-pnp-sweep/20-10-15-fetch-pnp--exp1-pnp-sweep_2020_10_16_00_39_25_id002--s956861/params.pt'
    # base_dir = '/home/vitchyr/mnt2/log2/paper-results/odac/fetch-pnp'
    # prefix = 'policy_only_'
    # alpha = 0.0042  # for fetch pnp
    # reward_scale = 0.0094  # for fetch pnp

    # snapshot_path = '/home/vitchyr/mnt2/log2/20-10-15-fetch-slide--exp1-slide-sweep/20-10-15-fetch-slide--exp1-slide-sweep_2020_10_16_00_43_08_id002--s525010/params.pt'
    # base_dir = '/home/vitchyr/mnt2/log2/paper-results/odac/fetch-slide'
    # alpha = 934e-6  # for fetch slide
    # reward_scale = 0.0058  # for fetch slide

    # snapshot_path = '/home/vitchyr/mnt2/log2/20-10-15-fetch-slide--exp1-slide-sweep/20-10-15-fetch-slide--exp1-slide-sweep_2020_10_16_00_43_08_id002--s525010/params.pt'
    # base_dir = '/home/vitchyr/mnt2/log2/paper-results/odac/fetch-slide'
    # alpha = 934e-6  # for fetch slide
    # reward_scale = 0.0058  # for fetch slide


    # snapshot_path = '/home/vitchyr/mnt2/log2/20-10-14-fetch--exp13-more-seeds-good-settings/20-10-14-fetch--exp13-more-seeds-good-settings_2020_10_14_07_51_12_id003--s901331/params.pt'
    # base_dir = '/home/vitchyr/mnt2/log2/paper-results/odac/fetch-push/dev'
    # alpha = 0.00178  # fetch push
    # reward_scale = 0.0153  # fetch push

    # snapshot_path = '/home/vitchyr/mnt2/log2/20-10-14-sawyer--exp5-discounted-kl-and-learned-model-laplace-global-variance/20-10-14-sawyer--exp5-discounted-kl-and-learned-model-laplace-global-variance_2020_10_14_18_37_39_id003--s519592/params.pt'
    # base_dir = '/home/vitchyr/mnt2/log2/paper-results/odac/sawyer-push/dev-ghost'
    # alpha = .00717  # sawyer-push
    # reward_scale = 0.495  # sawyer-push

    ptu.set_gpu_mode(True)
    data = torch.load(snapshot_path, map_location=ptu.device)
    # policy = data['evaluation/eval/policy']
    policy = data['exploration/policy']
    qf1 = data['trainer/qf1']
    qf2 = data['trainer/qf2']
    target_qf1 = data['trainer/target_qf1']
    target_qf2 = data['trainer/target_qf2']
    trainer_policy = data['trainer/policy']
    state_env = data['evaluation/eval/env']
    observation_key = data['evaluation/eval/observation_key']
    context_keys = data['evaluation/eval/context_keys_for_policy']

    reward_fn = state_env.reward_fn
    context_key = context_keys[0]

    renderer_keep_keys = [
        'image_observation',
        'discount_0.99',
        'reward',
        'bootstrap-value',
    ]

    plot_renderer_kwargs = {
        "dpi": 64,
        "height": 512,
        "output_image_format": "CHW",
        "width": 512
    }

    raw_rewards = []
    def get_reward(obs_dict, action, next_obs_dict, rs=1.):
        o = batchify(obs_dict)
        a = batchify(action)
        next_o = batchify(next_obs_dict)
        reward = reward_fn(o, a, next_o, next_o)
        final_reward = reward[0] * rs
        raw_rewards.append(final_reward)
        return final_reward

    def get_bootstrap_stats(obs, actions, next_obs, alpha):
        q1_pred = qf1(obs, actions)
        q2_pred = qf2(obs, actions)
        next_dist = trainer_policy(next_obs)
        new_next_actions, new_log_pi = next_dist.rsample_and_logprob()
        new_log_pi = new_log_pi.unsqueeze(-1)
        bootstrap_log_pi_term = - alpha * new_log_pi
        bootstrap_value = torch.min(
            target_qf1(next_obs, new_next_actions),
            target_qf2(next_obs, new_next_actions),
        ) + bootstrap_log_pi_term
        return bootstrap_value, q1_pred, q2_pred, bootstrap_log_pi_term

    raw_bootstraps = []
    def get_bootstrap(obs_dict, action, next_obs_dict, return_float=True):
        context_pt = ptu.from_numpy(obs_dict[context_key][None])
        o_pt = ptu.from_numpy(obs_dict[observation_key][None])
        next_o_pt = ptu.from_numpy(next_obs_dict[observation_key][None])
        action_torch = ptu.from_numpy(action[None])
        bootstrap, *_ = get_bootstrap_stats(
            torch.cat((o_pt, context_pt), dim=1),
            action_torch,
            torch.cat((next_o_pt, context_pt), dim=1),
            alpha=alpha,
        )
        if return_float:
            bootstrap = ptu.get_numpy(bootstrap)[0, 0]
        raw_bootstraps.append(bootstrap)
        return bootstrap

    def get_scaled_bootstrap(obs_dict, action, next_obs_dict):
        bootstrap = get_bootstrap(obs_dict, action, next_obs_dict)
        discount = get_discount(obs_dict, action, next_obs_dict)
        return bootstrap * discount

    def get_scaled_reward(obs_dict, action, next_obs_dict, rs=1):
        reward = get_reward(obs_dict, action, next_obs_dict, rs=rs)
        discount = get_discount(obs_dict, action, next_obs_dict)
        return reward * (1-discount)

    discounts = []
    def get_discount(obs_dict, action, next_obs_dict, prior_discount=0.99):
        bootstrap = get_bootstrap(obs_dict, action, next_obs_dict, return_float=False)
        reward_np = get_reward(obs_dict, action, next_obs_dict)
        reward = ptu.from_numpy(reward_np[None, None])
        discount = compute_discount_factor(
            bootstrap,
            reward,
            reward_scale=reward_scale,
            prior_discount=prior_discount,
        )
        if isinstance(discount, torch.Tensor):
            discount = ptu.get_numpy(discount)[0, 0]
        discounts.append(copy.deepcopy(discount))
        return np.clip(discount, a_min=1e-3, a_max=1)

    def create_modify_fn(title, set_params=None, scientific=True,):
        def modify(ax):
            ax.set_title(title)
            if set_params:
                ax.set(**set_params)
            if scientific:
                scaler = ScalarFormatter(useOffset=True)
                scaler.set_powerlimits((1, 1))
                ax.yaxis.set_major_formatter(scaler)
                ax.ticklabel_format(axis='y', style='sci')
        return modify

    def add_left_margin(fig):
        fig.subplots_adjust(left=0.2)

    # hd_renderer = RoboverseEnvRenderer(
    #     output_image_format='CHW',
    #     width=1024,
    #     height=1024,
    #     get_view_matrix_kwargs=dict(
    #         target_pos=[.6, 0, -0.3],
    #         distance=.55,
    #         yaw=90,
    #         pitch=-40,
    #         roll=0,
    #         up_axis_index=2,
    #     ),
    # )
    # policy.to(ptu.device)

    hd_renderer = GymEnvRenderer(
        width=512,
        height=512,
        output_image_format='CHW',
    )
    renderers = OrderedDict(
        image_observation=hd_renderer,
    )
    renderers['log_discount_0.99'] = DynamicNumberEnvRenderer(
        dynamic_number_fn=partial(get_discount, prior_discount=0.99),
        modify_ax_fn=create_modify_fn(
            title='log inferred discount (prior=0.99)',
            set_params=dict(
                yscale='log',
                ylim=[0.95, 1.],
            ),
            # scientific=False,
        ),
        modify_fig_fn=add_left_margin,
        # autoscale_y=False,
        **plot_renderer_kwargs)
    renderers['discount_0.99'] = DynamicNumberEnvRenderer(
        dynamic_number_fn=partial(get_discount, prior_discount=0.99),
        modify_ax_fn=create_modify_fn(
            title='inferred discount (prior=0.99)',
            set_params=dict(
                # yscale='log',
                ylim=[-0.05, 1.1],
            ),
            # scientific=False,
        ),
        modify_fig_fn=add_left_margin,
        # autoscale_y=False,
        **plot_renderer_kwargs)
    renderers['discount_0.5'] = DynamicNumberEnvRenderer(
        dynamic_number_fn=partial(get_discount, prior_discount=0.5),
        modify_ax_fn=create_modify_fn(
            title='inferred discount (prior=0.5)',
            set_params=dict(
                # yscale='log',
                ylim=[-0.05, 1.1],
            ),
            # scientific=False,
        ),
        modify_fig_fn=add_left_margin,
        # autoscale_y=False,
        **plot_renderer_kwargs)

    renderers['scaled-reward'] = DynamicNumberEnvRenderer(
        dynamic_number_fn=partial(get_scaled_reward, rs=reward_scale),
        modify_ax_fn=create_modify_fn(
            title='scaled reward',
        ),
        modify_fig_fn=add_left_margin,
        **plot_renderer_kwargs)
    renderers['scaled-bootstrap-value'] = DynamicNumberEnvRenderer(
        dynamic_number_fn=get_scaled_bootstrap,
        modify_ax_fn=create_modify_fn(
            title='scaled bootstrap value',
            # scientific=False,
        ),
        modify_fig_fn=add_left_margin,
        **plot_renderer_kwargs)
    renderers['reward'] = DynamicNumberEnvRenderer(
        dynamic_number_fn=partial(get_reward, rs=reward_scale),
        modify_ax_fn=create_modify_fn(
            title='reward',
            # scientific=False,
        ),
        modify_fig_fn=add_left_margin,
        **plot_renderer_kwargs)
    renderers['bootstrap-value'] = DynamicNumberEnvRenderer(
        dynamic_number_fn=get_bootstrap,
        modify_ax_fn=create_modify_fn(
            title='bootstrap value',
            # scientific=False,
        ),
        modify_fig_fn=add_left_margin,
        **plot_renderer_kwargs)

    if renderer_keep_keys:
        keys_to_del = [k for k in renderers if k not in renderer_keep_keys]
        for k in keys_to_del:
            renderers.pop(k)

    base_env = state_env.unwrapped
    is_gym_env = (
            isinstance(base_env, FetchEnv)
            or isinstance(base_env, AntXYGoalEnv)
            or isinstance(base_env, AntFullPositionGoalEnv)
    )

    def add_images(env):
        base_env = env.env
        base_distribution = env.context_distribution
        if is_gym_env:
            goal_distribution = base_distribution
        else:
            goal_distribution = AddImageDistribution(
                env=base_env,
                base_distribution=base_distribution,
                image_goal_key='image_desired_goal',
                renderer=hd_renderer,
            )
        context_env = ContextualEnv(
            base_env,
            context_distribution=goal_distribution,
            reward_fn=reward_fn,
            observation_key=observation_key,
            update_env_info_fn=delete_info,
        )
        return InsertDebugImagesEnv(
            context_env,
            renderers=renderers,
        )
    def set_goal_for_visualization(env, policy, o):
        goal = o['desired_goal']
        env.unwrapped.goal = goal
        goal[0] += 0.1
    if is_gym_env:
        keys = list(renderers.keys())
    else:
        keys = ['image_desired_goal'] + list(renderers.keys())
    img_env = add_images(state_env)
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    for i in range(num_imgs):
        rollout_function = partial(
            rf.contextual_rollout,
            observation_key=observation_key,
            context_keys_for_policy=context_keys,
            reset_callback=set_goal_for_visualization if is_gym_env else None,
        )
        dump_video(
            env=img_env,
            policy=policy,
            filename='{}/{}rollout{}.mp4'.format(base_dir, prefix, i),
            rollout_function=rollout_function,
            keys_to_show=keys,
            horizon=horizon,
            img_chw=hd_renderer.image_chw,
            image_format=hd_renderer.output_image_format,
            columns=1,
            rows=1,
            num_columns_per_rollout=len(keys),
            pad_length=0,
            subpad_length=0,
        )
        for subprefix, lst in [
            ('discount', discounts),
            ('raw_rewards', raw_rewards),
            ('raw_bootstraps', raw_bootstraps),
        ]:
            if len(lst) == 0:
                import ipdb; ipdb.set_trace()
            discounts_file = '{}/{}{}_rollout{}.npy'.format(
                base_dir, prefix, subprefix, i)
            with open(discounts_file, 'wb') as f:
                if isinstance(lst, torch.Tensor):
                    np.save(f, ptu.get_numpy(torch.cat(lst)).flatten())
                else:
                    np.save(f, np.array(lst).flatten())
            lst.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path')
    parser.add_argument('--save_dir')
    parser.add_argument('--H', default=100, type=int)
    parser.add_argument('--N', default=5, type=int)
    args = parser.parse_args()

    save_video(
        args.path,
        args.save_dir,
        horizon=args.H,
        num_imgs=args.N,
    )
