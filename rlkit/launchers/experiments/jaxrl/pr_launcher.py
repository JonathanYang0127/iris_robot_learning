def pr_experiment(variant):
    import os

    import numpy as np
    import tqdm
    import ml_collections
    from tensorboardX import SummaryWriter

    from jaxrl.agents import AWACLearner, SACLearner
    from learner import Learner
    from jaxrl.datasets import ReplayBuffer
    from jaxrl.datasets.dataset_utils import make_env_and_dataset
    from jaxrl.evaluation import evaluate
    from jaxrl.utils import make_env

    from rlkit.core import logger

    variant = ml_collections.ConfigDict(variant)
    kwargs = variant.config

    seed = variant.seedid # seed
    save_dir = logger.get_snapshot_dir()

    summary_writer = SummaryWriter(
        os.path.join(save_dir, 'tb', str(seed)))

    if variant.save_video:
        video_train_folder = os.path.join(save_dir, 'video', 'train')
        video_eval_folder = os.path.join(save_dir, 'video', 'eval')
    else:
        video_train_folder = None
        video_eval_folder = None

    env, dataset = make_env_and_dataset(variant.env_name, seed,
                                        variant.dataset_name, video_train_folder, )

    eval_env = make_env(variant.env_name, seed + 42, video_eval_folder, )

    np.random.seed(seed)

    kwargs = dict(variant.config)
    algo = kwargs.pop('algo')
    replay_buffer_size = kwargs.pop('replay_buffer_size')
    if algo == 'sac':
        agent = SACLearner(seed,
                           env.observation_space.sample()[np.newaxis],
                           env.action_space.sample()[np.newaxis], **kwargs)
    elif algo == 'awac':
        agent = AWACLearner(seed,
                            env.observation_space.sample()[np.newaxis],
                            env.action_space.sample()[np.newaxis], **kwargs)
    elif algo == 'pr':
        max_reward = max(0.0, np.max(dataset.rewards))
        agent = Learner(seed,
                            env.observation_space.sample()[np.newaxis],
                            env.action_space.sample()[np.newaxis], max_reward, **kwargs)
    else:
        raise NotImplementedError()

    action_dim = env.action_space.shape[0]
    replay_buffer = ReplayBuffer(env.observation_space, action_dim,
                                 replay_buffer_size or variant.max_steps)
    replay_buffer.initialize_with_dataset(dataset, variant.init_dataset_size)

    eval_returns = []
    observation, done = env.reset(), False

    # Use negative indices for pretraining steps.
    for i in tqdm.tqdm(range(1 - variant.num_pretraining_steps,
                             variant.max_steps + 1),
                       smoothing=0.1,
                       disable=not variant.tqdm):
        if i >= 1:
            action = agent.sample_actions(observation)
            next_observation, reward, done, info = env.step(action)

            if not done or 'TimeLimit.truncated' in info:
                mask = 1.0
            else:
                mask = 0.0

            replay_buffer.insert(observation, action, reward, mask,
                                 next_observation)
            observation = next_observation

            if done:
                observation, done = env.reset(), False
                for k, v in info['episode'].items():
                    summary_writer.add_scalar(f'training/{k}', v,
                                              info['total']['timesteps'])
        else:
            info = {}
            info['total'] = {'timesteps': i}

        batch = replay_buffer.sample(variant.batch_size)
        update_info = agent.update(batch)

        if i % variant.log_interval == 0:
            for k, v in update_info.items():
                if v.ndim == 0:
                    summary_writer.add_scalar(f'training/{k}', v, i)
                else:
                    summary_writer.add_histogram(f'training/{k}', v, i)
            summary_writer.flush()

        if i % variant.eval_interval == 0:
            eval_stats = evaluate(agent, eval_env, variant.eval_episodes)

            for k, v in eval_stats.items():
                summary_writer.add_scalar(f'evaluation/average_{k}s', v,
                                          info['total']['timesteps'])
            summary_writer.flush()

            eval_returns.append(
                (info['total']['timesteps'], eval_stats['return']))
            np.savetxt(os.path.join(save_dir, 'progress.csv'),
                       eval_returns,
                       fmt=['%d', '%.1f'],
                       delimiter=",", header="expl/num steps total,eval/Average Returns")