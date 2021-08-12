import rlkit.util.hyperparameter as hyp
from rlkit.launchers.arglauncher import run_variants

def main(variant):
    import os

    import numpy as np
    import tqdm
    import ml_collections
    from tensorboardX import SummaryWriter

    from jaxrl.agents import AWACLearner, SACLearner
    from jaxrl.datasets import ReplayBuffer
    from jaxrl.datasets.dataset_utils import make_env_and_dataset
    from jaxrl.evaluation import evaluate
    from jaxrl.utils import make_env

    from rlkit.core import logger

    import mj_envs

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
                                        variant.dataset_name, video_train_folder)

    eval_env = make_env(variant.env_name, seed + 42, video_eval_folder)

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
            np.savetxt(os.path.join(save_dir, 'progress.csv'),
                       eval_returns,
                       fmt=['%d', '%.1f'],
                       delimiter=",", header="expl/num steps total,eval/Average Returns")


if __name__ == "__main__":
    # noinspection PyTypeChecker

    variant = dict(
        env_name = "Ant-v2",
        dataset_name = "awac",
        # save_dir = "./tmp/",
        seedid = 0,
        eval_episodes = 10,
        log_interval = 1000,
        eval_interval = 10000,
        batch_size = 1024,
        max_steps = int(1e6),
        init_dataset_size = None,
        num_pretraining_steps = int(5e4),
        tqdm = True,
        save_video = False,
        config = dict(
            algo = 'awac',
            actor_optim_kwargs = dict(
                learning_rate = 3e-4,
                weight_decay = 1e-4,
            ),
            actor_hidden_dims = (256, 256, 256, 256),
            state_dependent_std = False,
            critic_lr = 3e-4,
            critic_hidden_dims = (256, 256),
            discount = 0.99,
            tau = 0.005,
            target_update_period = 1,
            beta = 2.0,
            num_samples = 1,
            replay_buffer_size = None,
        )
    )

    search_space = {
        'env_name': ["relocate-binary-v0", "door-binary-v0", "pen-binary-v0",],
        'seedid': range(3),
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
