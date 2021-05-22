from collections import OrderedDict

import numpy as np


from rlkit.core.timer import timer
from rlkit.core import logger
from rlkit.core.logging import add_prefix
from rlkit.torch.core import np_to_pytorch_batch
import rlkit.torch.pytorch_util as ptu
from rlkit.misc import eval_util


def _get_epoch_timings():
    times_itrs = timer.get_times()
    times = OrderedDict()
    for key in sorted(times_itrs):
        time = times_itrs[key]
        times['time/{} (s)'.format(key)] = time
    return times


class SimpleOfflineRlAlgorithm(object):
    def __init__(
            self,
            trainer,
            replay_buffer,
            batch_size,
            logging_period,
            num_batches,
    ):
        self.trainer = trainer
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.logging_period = logging_period

    def train(self):
        # first train only the Q function
        iteration = 0
        for i in range(self.num_batches):
            train_data = self.replay_buffer.random_batch(self.batch_size)
            train_data = np_to_pytorch_batch(train_data)
            obs = train_data['observations']
            next_obs = train_data['next_observations']
            train_data['observations'] = obs
            train_data['next_observations'] = next_obs
            self.trainer.train_from_torch(train_data)
            if i % self.logging_period == 0:
                stats_with_prefix = add_prefix(
                    self.trainer.eval_statistics, prefix="trainer/")
                self.trainer.end_epoch(iteration)
                iteration += 1
                logger.record_dict(stats_with_prefix)
                logger.dump_tabular(with_prefix=True, with_timestamp=False)


class OfflineMetaRLAlgorithm(object):
    def __init__(
            self,
            # main objects needed
            env,
            meta_replay_buffer,
            replay_buffer,
            task_embedding_replay_buffer,
            trainer,
            train_tasks,
            eval_tasks,
            # settings
            batch_size,
            logging_period,
            meta_batch_size,
            num_batches,
            task_embedding_batch_size,
            extra_eval_fns=(),
            checkpoint_frequency=25,
            use_meta_learning_buffer=False,
            video_saver=None,
    ):
        self.env = env
        self.trainer = trainer
        self.meta_replay_buffer = meta_replay_buffer
        self.replay_buffer = replay_buffer
        self.task_embedding_replay_buffer = task_embedding_replay_buffer
        self.batch_size = batch_size
        self.task_embedding_batch_size = task_embedding_batch_size
        self.num_batches = num_batches
        self.logging_period = logging_period
        self.train_tasks = train_tasks
        self.eval_tasks = eval_tasks
        self.meta_batch_size = meta_batch_size
        self._extra_eval_fns = extra_eval_fns
        self.checkpoint_frequency = checkpoint_frequency
        self.use_meta_learning_buffer = use_meta_learning_buffer
        self.video_saver = video_saver

    def train(self):
        # first train only the Q function
        iteration = 0
        timer.return_global_times = True
        timer.reset()
        for i in range(self.num_batches):
            if self.use_meta_learning_buffer:
                train_data = self.meta_replay_buffer.sample_meta_batch(
                    rl_batch_size=self.batch_size,
                    meta_batch_size=self.meta_batch_size,
                    embedding_batch_size=self.task_embedding_batch_size,
                )
                train_data = np_to_pytorch_batch(train_data)
            else:
                task_indices = np.random.choice(
                    self.train_tasks, self.meta_batch_size,
                )
                train_data = self.replay_buffer.sample_batch(
                    task_indices,
                    self.batch_size,
                )
                train_data = np_to_pytorch_batch(train_data)
                obs = train_data['observations']
                next_obs = train_data['next_observations']
                train_data['observations'] = obs
                train_data['next_observations'] = next_obs
                train_data['context'] = (
                    self.task_embedding_replay_buffer.sample_context(
                        task_indices,
                        self.task_embedding_batch_size,
                    ))
            timer.start_timer('train', unique=False)
            self.trainer.train_from_torch(train_data)
            timer.stop_timer('train')

            if i % self.logging_period == 0:

                if i % (self.logging_period*self.checkpoint_frequency) == 0:
                    params = self.get_epoch_snapshot(i)
                    logger.save_itr_params(i, params)

                stats_with_prefix = add_prefix(
                    self.trainer.eval_statistics, prefix="trainer/")
                self.trainer.end_epoch(iteration)
                logger.record_dict(stats_with_prefix)

                for fn in self._extra_eval_fns:
                    extra_stats = fn()
                    logger.record_dict(extra_stats)
                timer.start_timer('eval', unique=False)
                # TODO: evaluate during offline RL
                eval_stats = self._do_eval(i)
                timer.stop_timer('eval',)
                eval_stats_with_prefix = add_prefix(eval_stats, prefix="eval/")
                logger.record_dict(eval_stats_with_prefix)

                logger.record_tabular('iteration', iteration)
                logger.record_dict(_get_epoch_timings())
                try:
                    import os
                    import psutil
                    process = psutil.Process(os.getpid())
                    logger.record_tabular('RAM Usage (Mb)', int(process.memory_info().rss / 1000000))
                except ImportError:
                    pass
                logger.dump_tabular(with_prefix=True, with_timestamp=False)
                iteration += 1

    def get_epoch_snapshot(self, epoch):
        snapshot = {'epoch': epoch}
        for k, v in self.trainer.get_snapshot().items():
            snapshot['trainer/' + k] = v
        # snapshot['env'] = self.env
        # snapshot['env_sampler'] = self.sampler
        # snapshot['agent'] = self.agent
        # snapshot['exploration_agent'] = self.exploration_agent
        return snapshot

    def to(self, device):
        self.trainer.to(device)

    def _do_eval(self, num_step):
        stats = OrderedDict()
        max_path_length = 25
        for task_idx in self.eval_tasks:
            initial_context = None
            self.trainer.agent.clear_z()
            for trial_idx in range(3):
                path = rollout(
                    self.env,
                    self.trainer.agent,
                    task_idx,
                    max_path_length=max_path_length,
                    initial_context=initial_context
                )
                initial_context = path['context']
                path_return = path['rewards'].sum()
                if self.video_saver is not None:
                    self.video_saver(num_step, 'eval_task_{}_trial_{}'.format(task_idx, trial_idx), path)
                stats['eval_env_{}_trial_{}/AverageReturns'.format(task_idx, trial_idx)] = path_return
        return stats

    # def get_eval_statistics(self):
    #     ### train tasks
    #     # eval on a subset of train tasks for speed
    #     stats = OrderedDict()
    #     indices = np.random.choice(self.train_task_indices, len(self.eval_task_indices))
    #     for key, path_collector in self.path_collectors.item():
    #         paths = path_collector.collect_paths()
    #         returns = eval_util.get_average_returns(paths)
    #         stats[key + '/AverageReturns'] = returns
    #     return stats


def rollout(
        env,
        agent,
        task_idx,
        max_path_length=np.inf,
        accum_context=True,
        animated=False,
        save_frames=False,
        use_predicted_reward=False,
        resample_latent_period=0,
        update_posterior_period=0,
        initial_context=None,
        initial_reward_context=None,
        infer_posterior_at_start=True,
        initialized_z_reward=None,
):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos

    :param initial_context:
    :param infer_posterior_at_start: If True, infer the posterior from `initial_context` if possible.
    :param env:
    :param agent:
    :task_idx: the task index
    :param task_idx: the index of the task inside the environment.
    :param max_path_length:
    :param accum_context: if True, accumulate the collected context
    :param animated:
    :param save_frames: if True, save video of rollout
    :param resample_latent_period: How often to resample from the latent posterior, in units of env steps.
        If zero, never resample after the first sample.
    :param update_posterior_period: How often to update the latent posterior,
    in units of env steps.
        If zero, never update unless an initial context is provided, in which
        case only update at the start using that initial context.
    :return:
    """

    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    zs = []
    if initialized_z_reward is None:
        env.reset_task(task_idx)
    o = env.reset()
    next_o = None

    if animated:
        env.render()
    if initial_context is not None and len(initial_context) == 0:
        initial_context = None

    context = initial_context

    if infer_posterior_at_start and initial_context is not None:
        z_dist = agent.latent_posterior(context, squeeze=True)
    else:
        z_dist = agent.latent_prior

    if use_predicted_reward:
        if initialized_z_reward is None:
            z_reward_dist = agent.latent_posterior(
                initial_reward_context, squeeze=True, for_reward_prediction=True,
            )
            z_reward = ptu.get_numpy(z_reward_dist.sample())
        else:
            z_reward = initialized_z_reward

    z = ptu.get_numpy(z_dist.sample())
    for path_length in range(max_path_length):
        if resample_latent_period != 0 and path_length % resample_latent_period == 0:
            z = ptu.get_numpy(z_dist.sample())
        a, agent_info = agent.get_action(o, z)
        next_o, r, d, env_info = env.step(a)
        if use_predicted_reward:
            r = agent.infer_reward(o, a, z_reward)
            r = r[0]
        if accum_context:
            context = agent.update_context(
                context,
                [o, a, r, next_o, d, env_info],
            )
        # TODO: remove "context is not None" check after fixing first-loop hack
        if update_posterior_period != 0 and path_length % update_posterior_period == 0 and context is not None and len(context) > 0:
            z_dist = agent.latent_posterior(context, squeeze=True)
        zs.append(z)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        o = next_o
        if animated:
            env.render()
        if save_frames:
            from PIL import Image
            image = Image.fromarray(np.flipud(env.get_image()))
            env_info['frame'] = image
        env_infos.append(env_info)
        if d:
            break

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1 and not isinstance(observations[0], dict):
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.concatenate(
        (
            observations[1:, ...],
            np.expand_dims(next_o, 0)
        ),
        axis=0,
    )
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
        latents=np.array(zs),
        context=context,
    )
