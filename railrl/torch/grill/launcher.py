import os.path as osp
import pickle
import time

import cv2
import numpy as np

import railrl.torch.pytorch_util as ptu
from multiworld.core.image_env import ImageEnv
from railrl.core import logger
from railrl.data_management.obs_dict_replay_buffer import \
    ObsDictRelabelingBuffer
from railrl.data_management.online_vae_replay_buffer import \
    OnlineVaeRelabelingBuffer
from railrl.envs.vae_wrappers import VAEWrappedEnv, load_vae
from railrl.exploration_strategies.base import (
    PolicyWrappedWithExplorationStrategy
)
from railrl.exploration_strategies.epsilon_greedy import EpsilonGreedy
from railrl.exploration_strategies.gaussian_strategy import GaussianStrategy
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.misc.asset_loader import local_path_from_s3_or_local_path
from railrl.misc.ml_util import PiecewiseLinearSchedule
from railrl.state_distance.tdm_networks import TdmQf, TdmVf, TdmPolicy, StochasticTdmPolicy
from railrl.state_distance.tdm_td3 import TdmTd3
from railrl.state_distance.tdm_twin_sac import TdmTwinSAC
from railrl.torch.her.her_td3 import HerTd3
from railrl.torch.her.online_vae_her_td3 import OnlineVaeHerTd3
from railrl.torch.her.online_vae_joint_algo import OnlineVaeHerJointAlgo
from railrl.torch.networks import FlattenMlp, TanhMlpPolicy
from railrl.torch.td3.td3 import TD3
from railrl.torch.vae.conv_vae import ConvVAE, ConvVAETrainer


def grill_tdm_td3_full_experiment(variant):
    full_experiment_variant_preprocess(variant)
    generate_and_train_vae(variant)
    grill_tdm_td3_experiment(variant['grill_variant'])


def grill_tdm_twin_sac_full_experiment(variant):
    full_experiment_variant_preprocess(variant)
    generate_and_train_vae(variant)
    grill_tdm_twin_sac_experiment(variant['grill_variant'])


def grill_her_td3_full_experiment(variant):
    full_experiment_variant_preprocess(variant)
    generate_and_train_vae(variant)
    grill_her_td3_experiment(variant['grill_variant'])


def grill_her_td3_online_vae_full_experiment(variant):
    full_experiment_variant_preprocess(variant)
    generate_and_train_online_vae(variant)
    if variant['double_algo']:
        grill_her_td3_experiment_online_vae_exploring(variant['grill_variant'])
    else:
        grill_her_td3_experiment_online_vae(variant['grill_variant'])


def full_experiment_variant_preprocess(variant):
    train_vae_variant = variant['train_vae_variant']
    grill_variant = variant['grill_variant']
    env_class = variant['env_class']
    env_kwargs = variant['env_kwargs']
    init_camera = variant.get('init_camera', None)
    train_vae_variant['generate_vae_dataset_kwargs']['env_class'] = env_class
    train_vae_variant['generate_vae_dataset_kwargs']['env_kwargs'] = env_kwargs
    train_vae_variant['generate_vae_dataset_kwargs']['init_camera'] = init_camera
    grill_variant['env_class'] = env_class
    grill_variant['env_kwargs'] = env_kwargs
    grill_variant['init_camera'] = init_camera


def generate_and_train_vae(variant):
    grill_variant = variant['grill_variant']
    train_vae_variant = variant['train_vae_variant']
    if grill_variant.get('vae_path', None) is None:
        logger.remove_tabular_output(
            'progress.csv', relative_to_snapshot_dir=True
        )
        logger.add_tabular_output(
            'vae_progress.csv', relative_to_snapshot_dir=True
        )
        vae = train_vae(train_vae_variant)
        logger.save_extra_data(vae, 'vae.pkl', mode='pickle')
        logger.remove_tabular_output(
            'vae_progress.csv',
            relative_to_snapshot_dir=True,
        )
        logger.add_tabular_output(
            'progress.csv',
            relative_to_snapshot_dir=True,
        )
        grill_variant['vae_path'] = vae  # just pass the VAE directly


def generate_and_train_online_vae(variant):
    grill_variant = variant['grill_variant']
    train_vae_variant = variant['train_vae_variant']
    if grill_variant.get('vae_path', None) is None:
        logger.remove_tabular_output(
            'progress.csv', relative_to_snapshot_dir=True
        )
        logger.add_tabular_output(
            'vae_progress.csv', relative_to_snapshot_dir=True
        )
        vae, vae_train_data, vae_test_data = train_vae(train_vae_variant, return_data=True)
        grill_variant['vae_train_data'] = vae_train_data
        grill_variant['vae_test_data'] = vae_test_data
        logger.save_extra_data(vae, 'vae.pkl', mode='pickle')
        logger.remove_tabular_output(
            'vae_progress.csv',
            relative_to_snapshot_dir=True,
        )
        logger.add_tabular_output(
            'progress.csv',
            relative_to_snapshot_dir=True,
        )
        grill_variant['vae_path'] = vae  # just pass the VAE directly


def train_vae(variant, return_data=False):
    from railrl.core import logger
    import railrl.torch.pytorch_util as ptu
    beta = variant["beta"]
    representation_size = variant["representation_size"]
    train_data, test_data, info = generate_vae_dataset(
        **variant['generate_vae_dataset_kwargs']
    )
    logger.save_extra_data(info)
    logger.get_snapshot_dir()
    if 'beta_schedule_kwargs' in variant:
        beta_schedule = PiecewiseLinearSchedule(**variant['beta_schedule_kwargs'])
    else:
        beta_schedule = None
    m = ConvVAE(representation_size, **variant['vae_kwargs'])
    if ptu.gpu_enabled():
        m.cuda()
    t = ConvVAETrainer(train_data, test_data, m, beta=beta,
                       beta_schedule=beta_schedule, **variant['algo_kwargs'])
    save_period = variant['save_period']
    for epoch in range(variant['num_epochs']):
        should_save_imgs = (epoch % save_period == 0)
        t.train_epoch(epoch)
        t.test_epoch(
            epoch,
            save_reconstruction=should_save_imgs,
            save_scatterplot=should_save_imgs,
            # save_vae=False,
        )
        if should_save_imgs:
            t.dump_samples(epoch)
    logger.save_extra_data(m, 'vae.pkl', mode='pickle')
    if return_data:
        return m, train_data, test_data
    return m


def generate_vae_dataset(
        env_class,
        N=10000,
        test_p=0.9,
        use_cached=True,
        imsize=84,
        num_channels=1,
        show=False,
        init_camera=None,
        dataset_path=None,
        env_kwargs=None,
        oracle_dataset=False,
        n_random_steps=100,
):
    if env_kwargs is None:
        env_kwargs = {}
    filename = "/tmp/{}_{}_{}_oracle{}.npy".format(
        env_class.__name__,
        str(N),
        init_camera.__name__ if init_camera else '',
        oracle_dataset,
    )
    info = {}
    if dataset_path is not None:
        filename = local_path_from_s3_or_local_path(dataset_path)
        dataset = np.load(filename)
        N = dataset.shape[0]
    elif use_cached and osp.isfile(filename):
        dataset = np.load(filename)
        print("loaded data from saved file", filename)
    else:
        now = time.time()
        env = env_class(**env_kwargs)
        env = ImageEnv(
            env,
            imsize,
            init_camera=init_camera,
            transpose=True,
            normalize=True,
        )
        env.reset()
        info['env'] = env

        dataset = np.zeros((N, imsize * imsize * num_channels))
        for i in range(N):
            if oracle_dataset:
                goal = env.sample_goal()
                env.set_to_goal(goal)
            else:
                for _ in range(n_random_steps):
                    env.reset()
                    obs = env.step(env.action_space.sample())[0]
            obs = env.step(env.action_space.sample())[0]
            img = obs['image_observation']
            dataset[i, :] = img
            if show:
                img = img.reshape(3, 84, 84).transpose()
                img = img[::-1, :, ::-1]
                cv2.imshow('img', img)
                cv2.waitKey(1)
                # radius = input('waiting...')
        print("done making training data", filename, time.time() - now)
        np.save(filename, dataset)

    n = int(N * test_p)
    train_dataset = dataset[:n, :]
    test_dataset = dataset[n:, :]
    return train_dataset, test_dataset, info


def get_envs(variant):
    env = variant["env_class"](**variant['env_kwargs'])
    render = variant["render"]
    vae_path = variant.get("vae_path", None)
    reward_params = variant.get("reward_params", dict())
    init_camera = variant.get("init_camera", None)
    do_state_exp = variant.get("do_state_exp", False)
    if not do_state_exp:
        env = ImageEnv(
            env,
            84,
            init_camera=init_camera,
            transpose=True,
            normalize=True,
        )

        env = VAEWrappedEnv(
            env,
            vae_path,
            decode_goals=render,
            render_goals=render,
            render_rollouts=render,
            reward_params=reward_params,
            **variant.get('vae_wrapped_env_kwargs', {})
        )
    testing_env = pickle.loads(pickle.dumps(env))
    training_env = pickle.loads(pickle.dumps(env))
    relabeling_env = pickle.loads(pickle.dumps(env))
    video_vae_env = pickle.loads(pickle.dumps(env))
    video_goal_env = pickle.loads(pickle.dumps(env))
    if not do_state_exp:
        training_mode = variant.get("training_mode", "train")
        testing_mode = variant.get("testing_mode", "test")
        testing_env.mode(testing_mode)
        training_env.mode(training_mode)
        relabeling_env.mode(training_mode)
        relabeling_env.disable_render()
        video_vae_env.mode("video_vae")
        video_goal_env.mode("video_env")
    return testing_env, training_env, relabeling_env, video_vae_env, \
           video_goal_env


def get_exploration_strategy(variant, env):
    exploration_type = variant['exploration_type']
    exploration_noise = variant.get('exploration_noise', 0.1)
    if exploration_type == 'ou':
        es = OUStrategy(action_space=env.action_space)
    elif exploration_type == 'gaussian':
        es = GaussianStrategy(
            action_space=env.action_space,
            max_sigma=exploration_noise,
            min_sigma=exploration_noise,  # Constant sigma
        )
    elif exploration_type == 'epsilon':
        es = EpsilonGreedy(
            action_space=env.action_space,
            prob_random_action=exploration_noise,
        )
    else:
        raise Exception("Invalid type: " + exploration_type)
    return es


def grill_preprocess_variant(variant):
    if variant.get("do_state_exp", False):
        variant['observation_key'] = 'state_observation'
        variant['desired_goal_key'] = 'state_desired_goal'
        variant['achieved_goal_key'] = 'state_acheived_goal'


def grill_her_td3_experiment(variant):
    grill_preprocess_variant(variant)
    testing_env, training_env, relabeling_env, video_vae_env, video_goal_env = (
        get_envs(variant)
    )
    es = get_exploration_strategy(variant, training_env)

    observation_key = variant.get('observation_key', 'latent_observation')
    desired_goal_key = variant.get('desired_goal_key', 'latent_desired_goal')
    achieved_goal_key = desired_goal_key.replace("desired", "achieved")
    obs_dim = (
        training_env.observation_space.spaces[observation_key].low.size
        + training_env.observation_space.spaces[desired_goal_key].low.size
    )
    action_dim = training_env.action_space.low.size
    hidden_sizes = variant.get('hidden_sizes', [400, 300])
    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes,
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes,
    )
    policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        hidden_sizes=hidden_sizes,
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )

    replay_buffer = ObsDictRelabelingBuffer(
        env=relabeling_env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **variant['replay_kwargs']
    )
    variant["algo_kwargs"]["replay_buffer"] = replay_buffer
    render = variant["render"]
    algorithm = HerTd3(
        testing_env,
        training_env=training_env,
        qf1=qf1,
        qf2=qf2,
        policy=policy,
        exploration_policy=exploration_policy,
        render=render,
        render_during_eval=render,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        **variant['algo_kwargs']
    )

    if ptu.gpu_enabled():
        print("using GPU")
        algorithm.cuda()
        if not variant.get("do_state_exp", False):
            for e in [testing_env, training_env, video_vae_env, video_goal_env,
                      relabeling_env]:
                e.vae.cuda()

    save_video = variant.get("save_video", True)
    if save_video:
        from railrl.torch.vae.sim_vae_policy import dump_video
        logdir = logger.get_snapshot_dir()
        # Don't dump initial video any more, its uninformative
        # filename = osp.join(logdir, 'video_0_env.mp4')
        # dump_video(video_goal_env, policy, filename)
        # filename = osp.join(logdir, 'video_0_vae.mp4')
        # dump_video(video_vae_env, policy, filename)
    algorithm.train()

    if save_video:
        filename = osp.join(logdir, 'video_final_env.mp4')
        dump_video(video_goal_env, policy, filename)
        filename = osp.join(logdir, 'video_final_vae.mp4')
        dump_video(video_vae_env, policy, filename)


def grill_tdm_td3_experiment(variant):
    grill_preprocess_variant(variant)
    testing_env, training_env, relabeling_env, video_vae_env, video_goal_env = (
        get_envs(variant)
    )
    es = get_exploration_strategy(variant, training_env)
    observation_key = variant.get('observation_key', 'latent_observation')
    desired_goal_key = variant.get('desired_goal_key', 'latent_desired_goal')
    achieved_goal_key = desired_goal_key.replace("desired", "achieved")
    obs_dim = (
        training_env.observation_space.spaces[observation_key].low.size
    )
    goal_dim = (
        training_env.observation_space.spaces[desired_goal_key].low.size
    )
    action_dim = training_env.action_space.low.size

    vectorized = 'vectorized' in training_env.reward_type
    variant['algo_kwargs']['tdm_kwargs']['vectorized'] = vectorized

    norm_order = training_env.norm_order
    variant['algo_kwargs']['tdm_kwargs']['norm_order'] = norm_order

    vectorized_qf = variant['qf_kwargs'].get('vectorized', False)
    variant['qf_kwargs']['vectorized'] = vectorized or vectorized_qf
    variant['qf_kwargs']['norm_order'] = norm_order

    qf1 = TdmQf(
        env=training_env,
        observation_dim=obs_dim,
        goal_dim=goal_dim,
        action_dim=action_dim,
        **variant['qf_kwargs']
    )
    qf2 = TdmQf(
        env=training_env,
        observation_dim=obs_dim,
        goal_dim=goal_dim,
        action_dim=action_dim,
        **variant['qf_kwargs']
    )
    policy = TdmPolicy(
        env=training_env,
        observation_dim=obs_dim,
        goal_dim=goal_dim,
        action_dim=action_dim,
        **variant['policy_kwargs']
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )

    replay_buffer = ObsDictRelabelingBuffer(
        env=relabeling_env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        vectorized=vectorized,
        **variant['replay_kwargs']
    )
    render = variant["render"]
    variant["algo_kwargs"]["replay_buffer"] = replay_buffer
    algo_kwargs = variant['algo_kwargs']
    base_kwargs = algo_kwargs['base_kwargs']
    base_kwargs['training_env'] = training_env
    base_kwargs['render'] = render
    base_kwargs['render_during_eval'] = render
    tdm_kwargs = algo_kwargs['tdm_kwargs']
    tdm_kwargs['observation_key'] = observation_key
    tdm_kwargs['desired_goal_key'] = desired_goal_key
    algorithm = TdmTd3(
        testing_env,
        qf1=qf1,
        qf2=qf2,
        policy=policy,
        exploration_policy=exploration_policy,
        **variant['algo_kwargs']
    )

    if ptu.gpu_enabled():
        print("using GPU")
        algorithm.cuda()
        if not variant.get("do_state_exp", False):
            for e in [testing_env, training_env, video_vae_env, video_goal_env,
                      relabeling_env]:
                e.vae.cuda()

    save_video = variant.get("save_video", True)
    if save_video:
        from railrl.torch.vae.sim_vae_policy import dump_video
        logdir = logger.get_snapshot_dir()
        # Don't dump initial video any more, its uninformative
        # filename = osp.join(logdir, 'video_0_env.mp4')
        # dump_video(video_goal_env, policy, filename)
        # filename = osp.join(logdir, 'video_0_vae.mp4')
        # dump_video(video_vae_env, policy, filename)
    algorithm.train()

    if save_video:
        filename = osp.join(logdir, 'video_final_env.mp4')
        dump_video(video_goal_env, policy, filename)
        filename = osp.join(logdir, 'video_final_vae.mp4')
        dump_video(video_vae_env, policy, filename)


def grill_tdm_twin_sac_experiment(variant):
    grill_preprocess_variant(variant)
    testing_env, training_env, relabeling_env, video_vae_env, video_goal_env = (
        get_envs(variant)
    )
    observation_key = variant.get('observation_key', 'latent_observation')
    desired_goal_key = variant.get('desired_goal_key', 'latent_desired_goal')
    achieved_goal_key = desired_goal_key.replace("desired", "achieved")
    obs_dim = (
        training_env.observation_space.spaces[observation_key].low.size
    )
    goal_dim = (
        training_env.observation_space.spaces[desired_goal_key].low.size
    )
    action_dim = training_env.action_space.low.size

    vectorized = 'vectorized' in training_env.reward_type
    variant['algo_kwargs']['tdm_kwargs']['vectorized'] = vectorized

    norm_order = training_env.norm_order
    variant['algo_kwargs']['tdm_kwargs']['norm_order'] = norm_order

    # vectorized_qf = variant['qf_kwargs'].get('vectorized', False)
    variant['qf_kwargs']['vectorized'] = vectorized # vectorized or vectorized_qf
    variant['vf_kwargs']['vectorized'] = vectorized # vectorized or vectorized_qf
    variant['qf_kwargs']['norm_order'] = norm_order
    variant['vf_kwargs']['norm_order'] = norm_order

    qf1 = TdmQf(
        env=training_env,
        observation_dim=obs_dim,
        goal_dim=goal_dim,
        action_dim=action_dim,
        **variant['qf_kwargs']
    )
    qf2 = TdmQf(
        env=training_env,
        observation_dim=obs_dim,
        goal_dim=goal_dim,
        action_dim=action_dim,
        **variant['qf_kwargs']
    )
    vf = TdmVf(
        env=training_env,
        observation_dim=obs_dim,
        goal_dim=goal_dim,
        action_dim=action_dim,
        **variant['vf_kwargs']
    )
    policy = StochasticTdmPolicy(
        env=training_env,
        observation_dim=obs_dim,
        goal_dim=goal_dim,
        action_dim=action_dim,
        **variant['policy_kwargs']
    )

    replay_buffer = ObsDictRelabelingBuffer(
        env=relabeling_env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        vectorized=vectorized,
        **variant['replay_kwargs']
    )
    render = variant["render"]
    variant["algo_kwargs"]["replay_buffer"] = replay_buffer
    algo_kwargs = variant['algo_kwargs']
    base_kwargs = algo_kwargs['base_kwargs']
    base_kwargs['training_env'] = training_env
    base_kwargs['render'] = render
    base_kwargs['render_during_eval'] = render
    tdm_kwargs = algo_kwargs['tdm_kwargs']
    tdm_kwargs['observation_key'] = observation_key
    tdm_kwargs['desired_goal_key'] = desired_goal_key
    algorithm = TdmTwinSAC(
        testing_env,
        qf1=qf1,
        qf2=qf2,
        vf=vf,
        policy=policy,
        **variant['algo_kwargs']
    )

    if ptu.gpu_enabled():
        print("using GPU")
        algorithm.cuda()
        if not variant.get("do_state_exp", False):
            for e in [testing_env, training_env, video_vae_env, video_goal_env,
                      relabeling_env]:
                e.vae.cuda()

    save_video = variant.get("save_video", True)
    if save_video:
        from railrl.torch.vae.sim_vae_policy import dump_video
        logdir = logger.get_snapshot_dir()
        # Don't dump initial video any more, its uninformative
        # filename = osp.join(logdir, 'video_0_env.mp4')
        # dump_video(video_goal_env, policy, filename)
        # filename = osp.join(logdir, 'video_0_vae.mp4')
        # dump_video(video_vae_env, policy, filename)
    algorithm.train()

    if save_video:
        filename = osp.join(logdir, 'video_final_env.mp4')
        dump_video(video_goal_env, policy, filename)
        filename = osp.join(logdir, 'video_final_vae.mp4')
        dump_video(video_vae_env, policy, filename)


def grill_her_td3_experiment_online_vae(variant):
    grill_preprocess_variant(variant)
    testing_env, training_env, relabeling_env, video_vae_env, video_goal_env = (
        get_envs(variant)
    )
    es = get_exploration_strategy(variant, training_env)
    observation_key = variant.get('observation_key', 'latent_observation')
    desired_goal_key = variant.get('desired_goal_key', 'latent_desired_goal')
    achieved_goal_key = desired_goal_key.replace("desired", "achieved")
    obs_dim = (
        training_env.observation_space.spaces[observation_key].low.size
        + training_env.observation_space.spaces[desired_goal_key].low.size
    )
    action_dim = training_env.action_space.low.size
    hidden_sizes = variant.get('hidden_sizes', [400, 300])
    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes,
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes,
    )
    policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        hidden_sizes=hidden_sizes,
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )

    vae_path = variant["vae_path"]
    vae = load_vae(vae_path)

    replay_buffer = OnlineVaeRelabelingBuffer(
        vae=vae,
        env=relabeling_env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        exploration_rewards_type=variant.get('vae_exploration_rewards_type', 'None'),
        **variant['replay_kwargs']
    )
    variant["algo_kwargs"]["replay_buffer"] = replay_buffer

    t = ConvVAETrainer(variant['vae_train_data'],
                       variant['vae_test_data'],
                       vae,
                       beta=variant['online_vae_beta'])
    render = variant["render"]
    algorithm = OnlineVaeHerTd3(
        vae=vae,
        vae_trainer=t,
        env=testing_env,
        training_env=training_env,
        qf1=qf1,
        qf2=qf2,
        policy=policy,
        exploration_policy=exploration_policy,
        render=render,
        render_during_eval=render,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        vae_training_schedule=variant['vae_training_schedule'],
        **variant['algo_kwargs']
    )

    if ptu.gpu_enabled():
        print("using GPU")
        algorithm.cuda()
        vae.cuda()
    for e in [testing_env, training_env, video_vae_env, video_goal_env, relabeling_env]:
        e.vae = vae
        e.decode_goals = True

    save_video = variant.get("save_video", True)
    if save_video:
        from railrl.torch.vae.sim_vae_policy import dump_video
        logdir = logger.get_snapshot_dir()
        # Don't dump initial video any more, its uninformative
        # filename = osp.join(logdir, 'video_0_env.mp4')
        # dump_video(video_goal_env, policy, filename)
        # filename = osp.join(logdir, 'video_0_vae.mp4')
        # dump_video(video_vae_env, policy, filename)
    algorithm.train()

    if save_video:
        filename = osp.join(logdir, 'video_final_env.mp4')
        dump_video(video_goal_env, policy, filename)
        filename = osp.join(logdir, 'video_final_vae.mp4')
        dump_video(video_vae_env, policy, filename)


def grill_her_td3_experiment_online_vae_exploring(variant):
    grill_preprocess_variant(variant)
    testing_env, training_env, relabeling_env, video_vae_env, video_goal_env = (
        get_envs(variant)
    )
    es = get_exploration_strategy(variant, training_env)
    observation_key = variant.get('observation_key', 'latent_observation')
    desired_goal_key = variant.get('desired_goal_key', 'latent_desired_goal')
    achieved_goal_key = desired_goal_key.replace("desired", "achieved")
    obs_dim = (
        training_env.observation_space.spaces[observation_key].low.size
        + training_env.observation_space.spaces[desired_goal_key].low.size
    )
    action_dim = training_env.action_space.low.size
    hidden_sizes = variant.get('hidden_sizes', [400, 300])
    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes,
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes,
    )
    policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        hidden_sizes=hidden_sizes,
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )

    exploring_qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes,
    )
    exploring_qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes,
    )
    exploring_policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        hidden_sizes=hidden_sizes,
    )
    exploring_exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=exploring_policy,
    )

    vae_path = variant["vae_path"]
    vae = load_vae(vae_path)
    replay_buffer = OnlineVaeRelabelingBuffer(
        vae=vae,
        env=relabeling_env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        exploration_rewards_type=variant['vae_exploration_rewards_type'],
        **variant['replay_kwargs']
    )
    variant["algo_kwargs"]["replay_buffer"] = replay_buffer

    t = ConvVAETrainer(variant['vae_train_data'],
                       variant['vae_test_data'],
                       vae,
                       beta=variant['online_vae_beta'])

    control_algorithm = TD3(
        env=testing_env,
        training_env=training_env,
        qf1=qf1,
        qf2=qf2,
        policy=policy,
        exploration_policy=exploration_policy,
        **variant['algo_kwargs']
    )
    exploring_algorithm = TD3(
        env=testing_env,
        training_env=training_env,
        qf1=exploring_qf1,
        qf2=exploring_qf2,
        policy=exploring_policy,
        exploration_policy=exploring_exploration_policy,
        **variant['algo_kwargs']
    )

    algorithm = OnlineVaeHerJointAlgo(
        vae=vae,
        vae_trainer=t,
        env=testing_env,
        training_env=training_env,
        policy=policy,
        exploration_policy=exploration_policy,
        replay_buffer=replay_buffer,
        algo1=control_algorithm,
        algo2=exploring_algorithm,
        algo1_prefix="Control_",
        algo2_prefix="VAE_Exploration_",
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        vae_training_schedule=variant['vae_training_schedule'],
        **variant['joint_algo_kwargs']
    )

    if ptu.gpu_enabled():
        print("using GPU")
        algorithm.cuda()
        vae.cuda()
        for e in [testing_env, training_env, video_vae_env, video_goal_env, relabeling_env]:
            e.vae = vae
            e.decode_goals = True

    save_video = variant.get("save_video", True)
    if save_video:
        from railrl.torch.vae.sim_vae_policy import dump_video
        logdir = logger.get_snapshot_dir()
        # Don't dump initial video any more, its uninformative
        # filename = osp.join(logdir, 'video_0_env.mp4')
        # dump_video(video_goal_env, policy, filename)
        # filename = osp.join(logdir, 'video_0_vae.mp4')
        # dump_video(video_vae_env, policy, filename)
    algorithm.train()

    if save_video:
        filename = osp.join(logdir, 'video_final_env.mp4')
        dump_video(video_goal_env, policy, filename)
        filename = osp.join(logdir, 'video_final_vae.mp4')
        dump_video(video_vae_env, policy, filename)


