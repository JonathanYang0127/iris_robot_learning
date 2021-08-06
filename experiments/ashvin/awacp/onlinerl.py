import rlkit.util.hyperparameter as hyp
from rlkit.launchers.arglauncher import run_variants

def main(variant):
    import os
    import random

    import numpy as np
    import tqdm
    import ml_collections
    from tensorboardX import SummaryWriter

    from jaxrl.agents import AWACLearner, DDPGLearner, SACLearner, SACV1Learner
    from jaxrl.datasets import ReplayBuffer
    from jaxrl.evaluation import evaluate
    from jaxrl.utils import make_env
    import jaxrl
    import pathlib
    from rlkit.core import logger

    variant = ml_collections.ConfigDict(variant)
    kwargs = variant.config

    seed = 42 # variant.seed
    save_dir = logger.get_snapshot_dir()

    summary_writer = SummaryWriter(
        os.path.join(save_dir, 'tb', str(seed)))

    if variant.save_video:
        video_train_folder = os.path.join(save_dir, 'video', 'train')
        video_eval_folder = os.path.join(save_dir, 'video', 'eval')
    else:
        video_train_folder = None
        video_eval_folder = None

    env = make_env(variant.env_name, seed, video_train_folder)
    eval_env = make_env(variant.env_name, seed + 42, video_eval_folder)

    np.random.seed(seed)
    random.seed(seed)

    kwargs = dict(variant.config)
    algo = kwargs.pop('algo')
    replay_buffer_size = kwargs.pop('replay_buffer_size')
    if algo == 'sac':
        agent = SACLearner(seed,
                           env.observation_space.sample()[np.newaxis],
                           env.action_space.sample()[np.newaxis], **kwargs)
    elif algo == 'sac_v1':
        agent = SACV1Learner(seed,
                             env.observation_space.sample()[np.newaxis],
                             env.action_space.sample()[np.newaxis], **kwargs)
    elif algo == 'awac':
        agent = AWACLearner(seed,
                            env.observation_space.sample()[np.newaxis],
                            env.action_space.sample()[np.newaxis], **kwargs)
    elif algo == 'ddpg':
        agent = DDPGLearner(seed,
                            env.observation_space.sample()[np.newaxis],
                            env.action_space.sample()[np.newaxis], **kwargs)
    else:
        raise NotImplementedError()

    action_dim = env.action_space.shape[0]
    replay_buffer = ReplayBuffer(env.observation_space, action_dim,
                                 replay_buffer_size or variant.max_steps)

    eval_returns = []
    observation, done = env.reset(), False
    for i in tqdm.tqdm(range(1, variant.max_steps + 1),
                       smoothing=0.1,
                       disable=not variant.tqdm):
        if i < variant.start_training:
            action = env.action_space.sample()
        else:
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

            if 'is_success' in info:
                summary_writer.add_scalar(f'training/success',
                                          info['is_success'],
                                          info['total']['timesteps'])

        if i >= variant.start_training:
            batch = replay_buffer.sample(variant.batch_size)
            update_info = agent.update(batch)

            if i % variant.log_interval == 0:
                for k, v in update_info.items():
                    summary_writer.add_scalar(f'training/{k}', v, i)
                summary_writer.flush()

        if i % variant.eval_interval == 0:
            eval_stats = evaluate(agent, eval_env, variant.eval_episodes)

            for k, v in eval_stats.items():
                summary_writer.add_scalar(f'evaluation/average_{k}s', v,
                                          info['total']['timesteps'])
            summary_writer.flush()

            eval_returns.append(
                (info['total']['timesteps'], eval_stats['return']))
            np.savetxt(os.path.join(save_dir, f'{seed}.txt'),
                       eval_returns,
                       fmt=['%d', '%.1f'])


if __name__ == "__main__":
    # noinspection PyTypeChecker

    variant = dict(
        env_name = "HalfCheetah-v2",
        # save_dir = "./tmp/",
        seed = 42,
        eval_episodes = 10,
        log_interval = 1000,
        eval_interval = 5000,
        batch_size = 256,
        max_steps = int(1e6),
        start_training = int(1e4),
        tqdm = True,
        save_video = False,
        config = dict(
            algo = 'sac',
            actor_lr = 3e-4,
            critic_lr = 3e-4,
            temp_lr = 3e-4,
            hidden_dims = (256, 256),
            discount = 0.99,
            tau = 0.005,
            target_update_period = 1,
            init_temperature = 1.0,
            target_entropy = None,
            replay_buffer_size = None,
        )
    )

    search_space = {
        # 'env_name': [
        #     "antmaze-umaze-v0", "antmaze-umaze-diverse-v0", "antmaze-medium-play-v0",
        #     "antmaze-medium-diverse-v0", "antmaze-large-diverse-v0", "antmaze-large-play-v0",
        # ],
        # 'trainer_kwargs.lagrange_thresh': [5.0],
        # # 'trainer_kwargs.with_lagrange': [False],
        # 'trainer_kwargs.min_q_weight': [5.0, ],
        # 'seedid': range(3),
    }

    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    variants = []
    for variant in sweeper.iterate_hyperparameters():
        variants.append(variant)

    run_variants(main, variants, )

# if __name__ == '__main__':
#     app.run(main)
