from collections import OrderedDict

from rlkit.core.timer import timer
from rlkit.core.rl_algorithm import BaseRLAlgorithm
import numpy as np
from rlkit.core import logger
import os
from rlkit.exploration_strategies.embedding_wrappers import EmbeddingWrapperOffline, EmbeddingWrapper
import pickle

class BatchRLAlgorithmRnd(BaseRLAlgorithm):
    def __init__(
            self,
            perturb_trainer,
            batch_size,
            max_path_length,
            num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
            num_train_loops_per_epoch=1,
            min_num_steps_before_training=0,
            object_detector=None,
            multi_task=False,
            exploration_task=None,
            exploration_reset_task=None,
            exploration_task_prob=0.5,
            meta_batch_size=4,
            train_tasks=0,
            eval_tasks=0,
            biased_sampling=False,
            replay_buffer_positive=None,
            train_embedding_network=False,
            encoder_type='regular',
            save_exploration_paths=False,
            *args,
            **kwargs
    ):

        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training
        self.multi_task = multi_task
        self.exploration_task = exploration_task
        self.exploration_probability = exploration_task_prob
        self.exploration_reset_task = exploration_reset_task
        self.meta_batch_size = meta_batch_size
        self.train_tasks = train_tasks
        self.eval_tasks = eval_tasks
        self.object_detector = object_detector
        self.biased_sampling = biased_sampling
        self.train_embedding_network = train_embedding_network
        self.encoder_type = encoder_type
        self.save_exploration_paths = save_exploration_paths
        if self.train_embedding_network:
            assert replay_buffer_positive is not None
            self.replay_buffer_positive = replay_buffer_positive
        self.perturb_trainer = perturb_trainer

    def _train(self):
        done = (self.epoch == self.num_epochs)
        if done:
            return OrderedDict(), done

        if self.epoch == 0 and self.min_num_steps_before_training > 0:
            if self.multi_task:
                init_expl_paths = self.expl_data_collector.collect_new_paths(
                    self.max_path_length,
                    self.min_num_steps_before_training,
                    discard_incomplete_paths=True,
                    multi_task=True,
                    task_index=self.exploration_task
                )
                self.replay_buffer.add_multitask_paths(init_expl_paths)
            else:
                init_expl_paths = self.expl_data_collector.collect_new_paths(
                    self.max_path_length,
                    self.min_num_steps_before_training,
                    discard_incomplete_paths=False,
                )
                self.replay_buffer.add_paths(init_expl_paths)

            if self.save_exploration_paths:
                save_buffer_file = logger.get_snapshot_dir() + "/init_exploration_paths_{}.pkl".format(self.epoch)
                with open(save_buffer_file, "wb+") as f:
                    pickle.dump(init_expl_paths, f)
            self.expl_data_collector.end_epoch(-1)

        timer.start_timer('evaluation sampling')
        if self.epoch % self._eval_epoch_freq == 0 and self.num_eval_steps_per_epoch > 0:
            save_state_file = logger.get_snapshot_dir() + "/state{}.bullet".format(self.epoch)
            if (isinstance(self.eval_env, EmbeddingWrapper) or
                isinstance(self.eval_env, EmbeddingWrapperOffline)):
                self.eval_env.env.save_state(save_state_file)
            else:
                self.eval_env.save_state(save_state_file)
            if self.multi_task:
                for i in self.eval_tasks:
                    self.eval_data_collector.collect_new_paths(
                        self.max_path_length,
                        self.num_eval_steps_per_epoch,
                        discard_incomplete_paths=True,
                        multi_task=True,
                        task_index=i,
                    )
            else:
                self.eval_data_collector.collect_new_paths(
                    self.max_path_length,
                    self.num_eval_steps_per_epoch,
                    discard_incomplete_paths=True,
                )
            if (isinstance(self.eval_env, EmbeddingWrapper) or
                isinstance(self.eval_env, EmbeddingWrapperOffline)):
                self.eval_env.env.restore_state(save_state_file)
            else:
                self.eval_env.restore_state(save_state_file)
            os.remove(save_state_file)
        timer.stop_timer('evaluation sampling')

        if not self._eval_only:
            for _ in range(self.num_train_loops_per_epoch):
                if self.num_expl_steps_per_train_loop > 0:
                    if self.multi_task:
                        timer.start_timer('exploration sampling', unique=False)
                        print("collecting explorations")
                        new_expl_paths = self.expl_data_collector.collect_new_paths(
                            self.max_path_length,
                            self.num_expl_steps_per_train_loop,
                            discard_incomplete_paths=False,
                            multi_task=True,
                            task_index=self.exploration_task
                        )
                        print("done collecting explorations")
                        timer.stop_timer('exploration sampling')

                        timer.start_timer('replay buffer data storing', unique=False)
                        self.replay_buffer.add_multitask_paths(new_expl_paths)

                        if self.save_exploration_paths:
                            save_buffer_file = logger.get_snapshot_dir() + "/exploration_paths_{}.pkl".format(self.epoch)
                            with open(save_buffer_file, "wb+") as f:
                                pickle.dump(new_expl_paths, f)
                        timer.stop_timer('replay buffer data storing')
                    else:
                        timer.start_timer('exploration sampling', unique=False)
                        print("collecting explorations")
                        new_expl_paths = self.expl_data_collector.collect_new_paths(
                            self.max_path_length,
                            self.num_expl_steps_per_train_loop,
                            discard_incomplete_paths=False,
                            object_detector=self.object_detector,
                        )
                        print("done collecting explorations")
                        timer.stop_timer('exploration sampling')

                        timer.start_timer('replay buffer data storing', unique=False)
                        self.replay_buffer.add_paths(new_expl_paths)

                        timer.stop_timer('replay buffer data storing')
                        print("self.replay_buffer._size", self.replay_buffer._size)

                timer.start_timer('training', unique=False)
                for _ in range(self.num_trains_per_train_loop):
                    if not self.multi_task:
                        train_data = self.replay_buffer.random_batch(
                            self.batch_size, biased_sampling=self.biased_sampling)
                    else:
                        env_num_tasks = len(self.train_tasks) // 2
                        # only fwd task prior data and fwd expl data
                        task_indices_fwd = np.concatenate([self.train_tasks[:env_num_tasks], \
                                                          [self.exploration_task]])
                        # only perturb data
                        task_indices_perturb = [self.exploration_task + env_num_tasks]
                        task_indices_fwd = np.random.choice(
                            task_indices_fwd, self.meta_batch_size
                        )
                        task_indices_perturb = np.random.choice(
                            task_indices_perturb, self.meta_batch_size
                        )
                        try:
                            train_data_fwd = self.replay_buffer.sample_batch(
                                task_indices_fwd,
                                self.batch_size,
                            )
                            train_data_perturb = self.replay_buffer.sample_batch(
                                task_indices_perturb,
                                self.batch_size,
                            )
                        except:
                            import pdb; pdb.set_trace()
                            break
                    self.trainer.train(train_data_fwd)
                    self.perturb_trainer.train(train_data_perturb)
                timer.stop_timer('training')
        log_stats = self._get_diagnostics()
        return log_stats, False
