import abc
import pickle
import time
import gtimer as gt
import numpy as np
import ray

from railrl.data_management.env_replay_buffer import EnvReplayBuffer
from railrl.data_management.path_builder import PathBuilder
from railrl.data_management.split_buffer import SplitReplayBuffer
from railrl.envs.remote import RemoteRolloutEnv
from railrl.envs.wrappers import convert_gym_space
from railrl.policies.base import ExplorationPolicy
from railrl.samplers.in_place import InPlacePathSampler
from railrl.core import logger


class RLAlgorithm(metaclass=abc.ABCMeta):
    def __init__(
            self,
            env,
            exploration_policy: ExplorationPolicy,
            training_env=None,
            num_epochs=100,
            num_steps_per_epoch=10000,
            num_steps_per_eval=1000,
            num_updates_per_env_step=1,
            num_updates_per_epoch=None,
            batch_size=1024,
            max_path_length=1000,
            discount=0.99,
            replay_buffer_size=1000000,
            reward_scale=1,
            render=False,
            save_replay_buffer=False,
            save_algorithm=False,
            save_environment=True,
            eval_sampler=None,
            eval_policy=None,
            collection_mode='online',
            sim_throttle=False,
            normalize_env=True,
            env_train_ratio=20,
            parallel_step_to_train_ratio=20,
            replay_buffer=None,
            fraction_paths_in_train=1.,
    ):
        """
        Base class for RL Algorithms
        :param env: Environment used to evaluate.
        :param exploration_policy: Policy used to explore
        :param training_env: Environment used by the algorithm. By default, a
        copy of `env` will be made.
        :param num_epochs:
        :param num_steps_per_epoch:
        :param num_steps_per_eval:
        :param num_updates_per_env_step: Used by online training mode.
        :param num_updates_per_epoch: Used by batch training mode.
        :param batch_size:
        :param max_path_length:
        :param discount:
        :param replay_buffer_size:
        :param reward_scale:
        :param render:
        :param save_replay_buffer:
        :param save_algorithm:
        :param save_environment:
        :param eval_sampler:
        :param eval_policy: Policy to evaluate with.
        :param collection_mode:
        :param sim_throttle:
        :param normalize_env:
        :param parallel_step_to_train_ratio:
        :param replay_buffer:
        :param fraction_paths_in_train:
        """
        assert collection_mode in ['online', 'online-parallel', 'offline',
                                   'batch']
        assert 0. <= fraction_paths_in_train <= 1.
        if collection_mode == 'batch':
            assert num_updates_per_epoch is not None
        self.training_env = training_env or pickle.loads(pickle.dumps(env))
        self.normalize_env = normalize_env
        self.exploration_policy = exploration_policy
        self.num_epochs = num_epochs
        self.num_env_steps_per_epoch = num_steps_per_epoch
        self.num_steps_per_eval = num_steps_per_eval
        if collection_mode == 'online' or collection_mode == 'online-parallel':
            self.num_updates_per_train_call = num_updates_per_env_step
        else:
            self.num_updates_per_train_call = num_updates_per_epoch
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.discount = discount
        self.replay_buffer_size = replay_buffer_size
        self.reward_scale = reward_scale
        self.render = render
        self.collection_mode = collection_mode
        self.save_replay_buffer = save_replay_buffer
        self.save_algorithm = save_algorithm
        self.save_environment = save_environment
        if eval_sampler is None:
            if eval_policy is None:
                eval_policy = exploration_policy
            eval_sampler = InPlacePathSampler(
                env=env,
                policy=eval_policy,
                max_samples=self.num_steps_per_eval + self.max_path_length,
                max_path_length=self.max_path_length,
            )
        self.eval_policy = eval_policy
        self.eval_sampler = eval_sampler

        self.action_space = convert_gym_space(env.action_space)
        self.obs_space = convert_gym_space(env.observation_space)
        self.env = env
        if replay_buffer is None:
            if fraction_paths_in_train == 1.:
                self.replay_buffer = EnvReplayBuffer(
                    self.replay_buffer_size,
                    self.env,
                    flatten=True,
                )
            else:
                self.replay_buffer = SplitReplayBuffer(
                    EnvReplayBuffer(
                        replay_buffer_size,
                        env,
                        flatten=True,
                    ),
                    EnvReplayBuffer(
                        replay_buffer_size,
                        env,
                        flatten=True,
                    ),
                    fraction_paths_in_train=fraction_paths_in_train,
                )
        else:
            self.replay_buffer = replay_buffer
        self.replay_buffer_is_split = isinstance(
            self.replay_buffer,
            SplitReplayBuffer
        )

        self._n_env_steps_total = 0
        self._n_train_steps_total = 0
        self._n_rollouts_total = 0
        self._do_train_time = 0
        self._epoch_start_time = None
        self._algo_start_time = None
        self._old_table_keys = None
        self._current_path_builder = PathBuilder()
        self._exploration_paths = []
        self.parallel_step_to_train_ratio = parallel_step_to_train_ratio
        self.sim_throttle = sim_throttle
        if self.collection_mode == 'online-parallel':
            ray.init()
            self.training_env = RemoteRolloutEnv(
                env=env,
                policy=eval_policy,
                exploration_policy=exploration_policy,
                max_path_length=self.max_path_length,
                normalize_env=self.normalize_env,
            )

    def train(self, start_epoch=0):
        self.pretrain()
        if start_epoch == 0:
            params = self.get_epoch_snapshot(-1)
            logger.save_itr_params(-1, params)
        self.training_mode(False)
        self._n_env_steps_total = start_epoch * self.num_env_steps_per_epoch
        gt.reset()
        gt.set_def_unique(False)
        if self.collection_mode == 'online':
            self.train_online(start_epoch=start_epoch)
        elif self.collection_mode == 'online-parallel':
            self.train_parallel(start_epoch=start_epoch)
        elif self.collection_mode == 'batch':
            self.train_batch(start_epoch=start_epoch)
        elif self.collection_mode == 'offline':
            self.train_offline(start_epoch=start_epoch)
        else:
            raise TypeError("Invalid collection_mode: {}".format(
                self.collection_mode
            ))

    def pretrain(self):
        """
        Do anything before the main training phase.
        """
        pass

    def train_online(self, start_epoch=0):
        self._current_path_builder = PathBuilder()
        observation = self._start_new_rollout()
        for epoch in gt.timed_for(
                range(start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            self._start_epoch(epoch)
            for _ in range(self.num_env_steps_per_epoch):
                action, agent_info = self._get_action_and_info(
                    observation,
                )
                if self.render:
                    self.training_env.render()
                next_ob, raw_reward, terminal, env_info = (
                    self.training_env.step(action)
                )
                self._n_env_steps_total += 1
                reward = raw_reward * self.reward_scale
                terminal = np.array([terminal])
                reward = np.array([reward])
                self._handle_step(
                    observation,
                    action,
                    reward,
                    next_ob,
                    terminal,
                    agent_info=agent_info,
                    env_info=env_info,
                )
                if terminal or len(self._current_path_builder) >= self.max_path_length:
                    self._handle_rollout_ending()
                    observation = self._start_new_rollout()
                else:
                    observation = next_ob

                gt.stamp('sample')
                self._try_to_train()
                gt.stamp('train')

            self._try_to_eval(epoch)
            gt.stamp('eval')
            self._end_epoch()

    def train_batch(self, start_epoch):
        self._current_path_builder = PathBuilder()
        observation = self._start_new_rollout()
        for epoch in gt.timed_for(
                range(start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            self._start_epoch(epoch)
            for _ in range(self.num_env_steps_per_epoch):
                action, agent_info = self._get_action_and_info(
                    observation,
                )
                if self.render:
                    self.training_env.render()
                next_ob, raw_reward, terminal, env_info = (
                    self.training_env.step(action)
                )
                self._n_env_steps_total += 1
                reward = raw_reward * self.reward_scale
                terminal = np.array([terminal])
                reward = np.array([reward])
                self._handle_step(
                    observation,
                    action,
                    reward,
                    next_ob,
                    terminal,
                    agent_info=agent_info,
                    env_info=env_info,
                )
                if terminal or len(self._current_path_builder) >= self.max_path_length:
                    self._handle_rollout_ending()
                    observation = self._start_new_rollout()
                else:
                    observation = next_ob

            gt.stamp('sample')
            self._try_to_train()
            gt.stamp('train')

            self._try_to_eval(epoch)
            gt.stamp('eval')
            self._end_epoch()

    def train_parallel(self, start_epoch=0):
        assert isinstance(self.training_env, RemoteRolloutEnv), (
            "Did the sub-class accidentally override the RemoteRolloutEnv?"
        )
        self.training_mode(False)
        n_steps_current_epoch = 0
        epoch = start_epoch
        self._start_epoch(epoch)
        while self._n_env_steps_total <= self.num_epochs * self.num_env_steps_per_epoch:
            if self.sim_throttle:
                if epoch == 0 or self._n_env_steps_total//(self._n_train_steps_total+1) < self.parallel_step_to_train_ratio:
                    path = self.training_env.rollout(
                        self.exploration_policy,
                        use_exploration_strategy=True,
                    )
            else:
                path = self.training_env.rollout(
                    self.exploration_policy,
                    use_exploration_strategy=True,
                )
            if path is not None:
                path['rewards'] = path['rewards'] * self.reward_scale
                path_length = len(path['observations'])
                self._n_env_steps_total += path_length
                n_steps_current_epoch += path_length
                self._handle_path(path)
                if len(path) > 0:
                    self._exploration_paths.append(path)
            self._try_to_train()
            gt.stamp('sample')
            self._try_to_train()
            gt.stamp('train')
            # Check if epoch is over
            if n_steps_current_epoch >= self.num_env_steps_per_epoch:
                self._try_to_eval(epoch)
                gt.stamp('eval')
                self._end_epoch()
                epoch += 1
                n_steps_current_epoch = 0
                self._start_epoch(epoch)

    def train_offline(self, start_epoch=0):
        self.training_mode(False)
        params = self.get_epoch_snapshot(-1)
        logger.save_itr_params(-1, params)
        for epoch in range(start_epoch, self.num_epochs):
            self._start_epoch(epoch)
            self._try_to_train()
            self._try_to_offline_eval(epoch)
            self._end_epoch()

    def _try_to_train(self):
        if self._can_train():
            self.training_mode(True)
            for i in range(self.num_updates_per_train_call):
                self._do_training()
                self._n_train_steps_total += 1
            self.training_mode(False)

    def _try_to_eval(self, epoch):
        logger.save_extra_data(self.get_extra_data_to_save(epoch))
        if self._can_evaluate():
            self.evaluate(epoch)

            params = self.get_epoch_snapshot(epoch)
            logger.save_itr_params(epoch, params)
            table_keys = logger.get_table_key_set()
            if self._old_table_keys is not None:
                assert table_keys == self._old_table_keys, (
                    "Table keys cannot change from iteration to iteration."
                )
            self._old_table_keys = table_keys

            logger.record_tabular(
                "Number of train steps total",
                self._n_train_steps_total,
            )
            logger.record_tabular(
                "Number of env steps total",
                self._n_env_steps_total,
            )
            logger.record_tabular(
                "Number of rollouts total",
                self._n_rollouts_total,
            )

            if self.collection_mode != 'online-parallel':
                times_itrs = gt.get_times().stamps.itrs
                train_time = times_itrs['train'][-1]
                sample_time = times_itrs['sample'][-1]
                eval_time = times_itrs['eval'][-1] if epoch > 0 else 0
                epoch_time = train_time + sample_time + eval_time
                total_time = gt.get_times().total

                logger.record_tabular('Train Time (s)', train_time)
                logger.record_tabular('(Previous) Eval Time (s)', eval_time)
                logger.record_tabular('Sample Time (s)', sample_time)
                logger.record_tabular('Epoch Time (s)', epoch_time)
                logger.record_tabular('Total Train Time (s)', total_time)

            logger.record_tabular("Epoch", epoch)
            logger.dump_tabular(with_prefix=False, with_timestamp=False)
        else:
            logger.log("Skipping eval for now.")

    def _try_to_offline_eval(self, epoch):
        start_time = time.time()
        self.offline_evaluate(epoch)
        params = self.get_epoch_snapshot(epoch)
        logger.save_itr_params(epoch, params)
        table_keys = logger.get_table_key_set()
        if self._old_table_keys is not None:
            assert table_keys == self._old_table_keys, (
                "Table keys cannot change from iteration to iteration."
            )
        self._old_table_keys = table_keys
        logger.dump_tabular(with_prefix=False, with_timestamp=False)
        logger.log("Eval Time: {0}".format(time.time() - start_time))

    def _can_evaluate(self):
        """
        One annoying thing about the logger table is that the keys at each
        iteration need to be the exact same. So unless you can compute
        everything, skip evaluation.

        A common example for why you might want to skip evaluation is that at
        the beginning of training, you may not have enough data for a
        validation and training set.

        :return:
        """
        return (
            len(self._exploration_paths) > 0
            and self.replay_buffer.num_steps_can_sample() >= self.batch_size
        )

    def _can_train(self):
        return self.replay_buffer.num_steps_can_sample() >= self.batch_size

    def _get_action_and_info(self, observation):
        """
        Get an action to take in the environment.
        :param observation:
        :return:
        """
        self.exploration_policy.set_num_steps_total(self._n_env_steps_total)
        return self.exploration_policy.get_action(
            observation,
        )

    def _start_epoch(self, epoch):
        self._epoch_start_time = time.time()
        self._exploration_paths = []
        self._do_train_time = 0
        logger.push_prefix('Iteration #%d | ' % epoch)

    def _end_epoch(self):
        logger.log("Epoch Duration: {0}".format(
            time.time() - self._epoch_start_time
        ))
        logger.log("Started Training: {0}".format(self._can_train()))
        logger.pop_prefix()

    def _start_new_rollout(self):
        self.exploration_policy.reset()
        return self.training_env.reset()

    def _handle_path(self, path):
        """
        Naive implementation: just loop through each transition.
        :param path:
        :return:
        """
        for (
            ob,
            action,
            reward,
            next_ob,
            terminal,
            agent_info,
            env_info
        ) in zip(
            path["observations"],
            path["actions"],
            path["rewards"],
            path["next_observations"],
            path["terminals"],
            path["agent_infos"],
            path["env_infos"],
        ):
            self._handle_step(
                ob,
                action,
                reward,
                next_ob,
                terminal,
                agent_info=agent_info,
                env_info=env_info,
            )
        self._handle_rollout_ending()

    def _handle_step(
            self,
            observation,
            action,
            reward,
            next_observation,
            terminal,
            agent_info,
            env_info,
    ):
        """
        Implement anything that needs to happen after every step
        :return:
        """
        self._current_path_builder.add_all(
            observations=observation,
            actions=action,
            rewards=reward,
            next_observations=next_observation,
            terminals=terminal,
            agent_infos=agent_info,
            env_infos=env_info,
        )
        self.replay_buffer.add_sample(
            observation=observation,
            action=action,
            reward=reward,
            terminal=terminal,
            next_observation=next_observation,
            agent_info=agent_info,
            env_info=env_info,
        )

    def _handle_rollout_ending(self):
        """
        Implement anything that needs to happen after every rollout.
        """
        self.replay_buffer.terminate_episode()
        self._n_rollouts_total += 1
        if len(self._current_path_builder) > 0:
            self._exploration_paths.append(
                self._current_path_builder.get_all_stacked()
            )
            self._current_path_builder = PathBuilder()

    def get_epoch_snapshot(self, epoch):
        if self.render:
            self.training_env.render(close=True)
        data_to_save = dict(
            epoch=epoch,
            exploration_policy=self.exploration_policy,
        )
        if self.save_environment:
            data_to_save['env'] = self.training_env
        return data_to_save

    def get_extra_data_to_save(self, epoch):
        """
        Save things that shouldn't be saved every snapshot but rather
        overwritten every time.
        :param epoch:
        :return:
        """
        if self.render:
            self.training_env.render(close=True)
        data_to_save = dict(
            epoch=epoch,
        )
        if self.save_environment:
            data_to_save['env'] = self.training_env
        if self.save_replay_buffer:
            data_to_save['replay_buffer'] = self.replay_buffer
        if self.save_algorithm:
            data_to_save['algorithm'] = self
        return data_to_save

    @abc.abstractmethod
    def training_mode(self, mode):
        """
        Set training mode to `mode`.
        :param mode: If True, training will happen (e.g. set the dropout
        probabilities to not all ones).
        """
        pass

    @abc.abstractmethod
    def cuda(self):
        """
        Turn cuda on.
        :return:
        """
        pass

    @abc.abstractmethod
    def evaluate(self, epoch):
        """
        Evaluate the policy, e.g. save/print progress.
        :param epoch:
        :return:
        """
        pass

    # Don't make this abstract so that every class doesn't have to implement
    # this.
    def offline_evaluate(self, epoch):
        """
        Evaluate without collecting new data.
        :param epoch:
        :return:
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _do_training(self):
        """
        Perform some update, e.g. perform one gradient step.
        :return:
        """
        pass
