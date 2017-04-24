from collections import OrderedDict

import numpy as np
import tensorflow as tf

from railrl.algos.bptt_ddpg import BpttDDPG
from railrl.core import tf_util
from railrl.misc.data_processing import create_stats_ordered_dict
from railrl.pythonplusplus import filter_recursive, line_logger
from railrl.qfunctions.memory.oracle_unroll_qfunction import (
    OracleUnrollQFunction
)
from rllab.misc import special


class RegressQBpttDdpg(BpttDDPG):
    """
    Train the Q function by regressing onto the oracle Q values.
    """

    def __init__(
            self,
            *args,
            oracle_qf: OracleUnrollQFunction,
            qf_total_loss_tolerance=None,
            max_num_q_updates=100,
            env_grad_distance_weight=1.,
            write_grad_distance_weight=1.,
            qf_grad_mse_from_one_weight=1.,
            extra_train_period=100,
            regress_onto_values=False,
            num_extra_qf_updates=0,
            extra_qf_training_mode='none',
            validation_batch_size=None,
            **kwargs
    ):
        """

        :param args:
        :param oracle_qf:
        :param qf_total_loss_tolerance:
        :param max_num_q_updates:
        :param train_policy:
        :param env_grad_distance_weight:
        :param write_grad_distance_weight:
        :param qf_grad_mse_from_one_weight:
        :param extra_train_period:
        :param regress_onto_values:
        :param num_extra_qf_updates:
        :param extra_qf_training_mode: String:
         - 'none' : Don't do any extra QF training
         - 'fixed': Always do `num_extra_qf_updates` extra updates
         - 'validation': Do up to `max_num_q_updates` extra updates so long
         as validation qf loss goes down.
        :param kwargs:
        """
        assert extra_qf_training_mode in [
            'none',
            'fixed',
            'validation',
        ]
        super().__init__(*args, **kwargs)
        self.extra_qf_training_mode = extra_qf_training_mode
        self.qf_total_loss_tolerance = qf_total_loss_tolerance
        self.oracle_qf = oracle_qf
        self.max_num_q_updates = max_num_q_updates
        self.env_grad_distance_weight = env_grad_distance_weight
        self.write_grad_distance_weight = write_grad_distance_weight
        self.qf_grad_mse_from_one_weight = qf_grad_mse_from_one_weight
        self.extra_train_period = extra_train_period
        self.regress_onto_values = regress_onto_values
        self._num_extra_qf_updates = num_extra_qf_updates

        self._validation_batch_size = validation_batch_size or self.batch_size

    def _do_training(self, **kwargs):
        self._do_extra_qf_training(**kwargs)
        super()._do_training(**kwargs)

    def _do_extra_qf_training(self, n_steps_total=None, **kwargs):
        if self.extra_qf_training_mode == 'none':
            return
        elif self.extra_qf_training_mode == 'fixed':
            for _ in range(self._num_extra_qf_updates):
                minibatch = self._sample_minibatch()
                feed_dict = self._update_feed_dict_from_batch(minibatch)
                ops = filter_recursive([
                    self.train_qf_op,
                    self.update_target_qf_op,
                ])
                if len(ops) > 0:
                    self.sess.run(ops, feed_dict=feed_dict)
        elif self.extra_qf_training_mode == 'validation':
            if self.max_num_q_updates <= 0:
                return

            last_validation_loss = self._validation_qf_loss()
            if self.should_train_qf_extra(n_steps_total=n_steps_total):
                line_logger.print_over(
                    "{0} T:{1:3.4f} V:{2:3.4f}".format(0, 0, 0)
                )
                for i in range(self.max_num_q_updates):
                    minibatch = self._sample_minibatch()
                    feed_dict = self._update_feed_dict_from_batch(minibatch)
                    ops = [self.qf_loss, self.train_qf_op]
                    ops += self.update_target_qf_op
                    train_loss = float(
                        self.sess.run(ops, feed_dict=feed_dict)[0]
                    )
                    validation_loss = self._validation_qf_loss()
                    line_logger.print_over(
                        "{0} T:{1:3.4f} V:{2:3.4f}".format(
                            i, train_loss, validation_loss,
                        )
                    )
                    if validation_loss > last_validation_loss:
                        break
                    if validation_loss <= self.qf_total_loss_tolerance:
                        break
                    last_validation_loss = validation_loss
                line_logger.newline()

    def _validation_qf_loss(self):
        batch = self.pool.get_valid_subtrajectories(validation=True)
        feed_dict = self._update_feed_dict_from_batch(batch)
        return self.sess.run(self.qf_loss, feed_dict=feed_dict)

    def _init_policy_ops(self):
        super()._init_policy_ops()
        self.oracle_qf = self.oracle_qf.get_weight_tied_copy(
            action_input=self._final_rnn_action,
        )

    def _init_qf_loss_and_train_ops(self):
        # Compute a bunch of numbers that *could* be added to the loss.
        self.oracle_memory_grad = []
        self.oracle_env_grad = []
        self.env_grad = []
        self.memory_grad = []
        for env_action, memory_action in self._rnn_outputs:
            self.oracle_memory_grad += tf.gradients(self.oracle_qf.output,
                                                    memory_action)
            self.oracle_env_grad += tf.gradients(self.oracle_qf.output,
                                                 env_action)
            self.env_grad += tf.gradients(self.qf_with_action_input.output,
                                          env_action)
            self.memory_grad += tf.gradients(self.qf_with_action_input.output,
                                             memory_action)
        self.oracle_memory_grad = filter_recursive(self.oracle_memory_grad)
        self.oracle_memory_grad = tf.reduce_sum(self.oracle_memory_grad, axis=0)
        self.oracle_env_grad = tf.reduce_sum(self.oracle_env_grad, axis=0)
        self.env_grad = tf.reduce_sum(self.env_grad, axis=0)
        self.memory_grad = tf.reduce_sum(self.memory_grad, axis=0)

        self.mem_grad_cosine_distance = tf_util.cosine(self.oracle_memory_grad,
                                                       self.memory_grad)
        self.env_grad_cosine_distance = tf_util.cosine(self.oracle_env_grad,
                                                       self.env_grad)
        self.mem_grad_mse = tf_util.mse(self.oracle_memory_grad,
                                        self.memory_grad, axis=1)
        self.env_grad_mse = tf_util.mse(self.oracle_env_grad,
                                        self.env_grad, axis=1)
        self.env_qf_grad_mse_from_one = tf.reduce_mean(
            (tf.abs(self.env_grad) - 1) ** 2
        )
        self.memory_qf_grad_mse_from_one = tf.reduce_mean(
            (tf.abs(self.memory_grad) - 1) ** 2
        )
        oracle_qf_output = tf.expand_dims(self.oracle_qf.output, axis=1)
        self.true_qf_mse_loss = tf.squeeze(tf_util.mse(
            oracle_qf_output,
            self.qf.output,
        ))

        # The glorious if-then statement that maybe adds these numbers
        self.qf_loss = self.qf_weight_decay * self.Q_weights_norm
        if self.regress_onto_values:
            self.qf_loss += self.true_qf_mse_loss
        else:
            self.qf_loss += self.bellman_error

        if self.env_grad_distance_weight > 0.:
            self.qf_loss += - (
                tf.reduce_mean(self.env_grad_cosine_distance) *
                self.env_grad_distance_weight
            )
        if self.write_grad_distance_weight > 0.:
            self.qf_loss += - (
                tf.reduce_mean(self.mem_grad_cosine_distance) *
                self.write_grad_distance_weight
            )
        if self.qf_grad_mse_from_one_weight > 0.:
            self.qf_loss += (
                self.env_qf_grad_mse_from_one
                + self.memory_qf_grad_mse_from_one
            ) * self.qf_grad_mse_from_one_weight
        with tf.variable_scope("regress_train_qf_op"):
            self.train_qf_op = tf.train.AdamOptimizer(
                self.qf_learning_rate
            ).minimize(
                self.qf_loss,
                var_list=self.qf.get_params(),
            )

    def _qf_feed_dict(self, *args, **kwargs):
        feed_dict = super()._qf_feed_dict(*args, **kwargs)
        feed_dict.update(self._oracle_qf_feed_dict(*args, **kwargs))
        return feed_dict

    def _oracle_qf_feed_dict(self, rewards, terminals, obs, actions, next_obs,
                             target_numbers=None, times=None):
        batch_size = len(rewards)
        sequence_lengths = np.squeeze(self.env.horizon - 1 - times[:, -1])
        indices = target_numbers[:, 0]
        target_one_hots = special.to_onehot_n(
            indices,
            self.env.wrapped_env.action_space.flat_dim,
        )
        rest_of_obs = np.zeros(
            [
                batch_size,
                self.env.horizon - self._num_bptt_unrolls,
                self._env_obs_dim,
            ]
        )
        rest_of_obs[:, :, 0] = 1
        feed_dict = {
            self.oracle_qf.sequence_length_placeholder: sequence_lengths,
            self.oracle_qf.rest_of_obs_placeholder: rest_of_obs,
            self.oracle_qf.observation_input: obs,
            self.policy.observation_input: obs,
            self.oracle_qf.target_labels: target_one_hots,
        }
        if hasattr(self.qf, "target_labels"):
            feed_dict[self.qf.target_labels] = target_one_hots
            feed_dict[self.target_qf.target_labels] = target_one_hots
        if hasattr(self.qf, "time_labels"):
            feed_dict[self.qf.time_labels] = times[:, -1]
            feed_dict[self.target_qf.time_labels] = times[:, -1]
        return feed_dict

    def _update_feed_dict_from_batch(self, batch):
        return self._update_feed_dict(
            rewards=batch['rewards'],
            terminals=batch['terminals'],
            obs=batch['observations'],
            actions=batch['actions'],
            next_obs=batch['next_observations'],
            target_numbers=batch['target_numbers'],
            times=batch['times'],
        )

    def _statistic_names_and_ops(self):
        """
        :return: List of tuple (name, op). Each `op` will be evaluated. Its
        output will be saved as a statistic with name `name`.
        """
        return [
            ('PolicySurrogateLoss', self.policy_surrogate_loss),
            ('QfLoss', self.qf_loss),
            ('PolicyOutput', self.policy.output),
            ('QfOutput', self.qf.output),
            ('OracleQfOutput', self.oracle_qf.output),
        ]

    def _get_other_statistics(self):
        if self.pool.num_can_sample(validation=True) < self.batch_size:
            return {}

        statistics = OrderedDict()
        for name, validation in [
            ('Valid', True),
            ('Train', False),
        ]:
            batch = self.pool.get_valid_subtrajectories(validation=validation)
            feed_dict = self._update_feed_dict_from_batch(batch)
            (
                true_qf_mse_loss,
                bellman_error,
                qf_loss,
                env_grad_distance,
                memory_grad_distance,
                env_grad_mse,
                memory_grad_mse,
                env_qf_grad_mse_from_one,
                memory_qf_grad_mse_from_one,
                env_qf_grad,
                memory_qf_grad,
                oracle_env_qf_grad,
                oracle_memory_qf_grad,
                qf_output,
                oracle_qf_output,
            ) = self.sess.run(
                [
                    self.true_qf_mse_loss,
                    self.bellman_error,
                    self.qf_loss,
                    self.env_grad_cosine_distance,
                    self.mem_grad_cosine_distance,
                    self.env_grad_mse,
                    self.mem_grad_mse,
                    self.env_qf_grad_mse_from_one,
                    self.memory_qf_grad_mse_from_one,
                    self.env_grad,
                    self.memory_grad,
                    self.oracle_env_grad,
                    self.oracle_memory_grad,
                    self.qf.output,
                    self.oracle_qf.output,
                ]
                ,
                feed_dict=feed_dict
            )
            stat_base_name = 'Qf{}'.format(name)
            statistics.update(
                {'{}_True_MSE_Loss'.format(stat_base_name): true_qf_mse_loss},
            )
            statistics.update(
                {'{}_BellmanError'.format(stat_base_name): bellman_error},
            )
            statistics.update(
                {'{}_Loss'.format(stat_base_name): qf_loss},
            )
            statistics.update(create_stats_ordered_dict(
                '{}_Grad_Dist_env'.format(stat_base_name),
                env_grad_distance,
            ))
            statistics.update(create_stats_ordered_dict(
                '{}_Grad_Dist_memory'.format(stat_base_name),
                memory_grad_distance
            ))
            # statistics.update(create_stats_ordered_dict(
            #     '{}_Grad_MSE_env'.format(stat_base_name),
            #     env_grad_mse,
            # ))
            # statistics.update(create_stats_ordered_dict(
            #     '{}_Grad_MSE_memory'.format(stat_base_name),
            #     memory_grad_mse
            # ))
            # statistics.update(create_stats_ordered_dict(
            #     '{}_GradMSE_from_1_env'.format(stat_base_name),
            #     env_qf_grad_mse_from_one
            # ))
            # statistics.update(create_stats_ordered_dict(
            #     '{}_GradMSE_from_1_memory'.format(stat_base_name),
            #     memory_qf_grad_mse_from_one
            # ))
            # statistics.update(create_stats_ordered_dict(
            #     '{}_QF_Grads_env'.format(stat_base_name),
            #     env_qf_grad
            # ))
            # statistics.update(create_stats_ordered_dict(
            #     '{}_QF_Grads_memory'.format(stat_base_name),
            #     memory_qf_grad
            # ))
            # statistics.update(create_stats_ordered_dict(
            #     '{}_OracleQF_Grads_env'.format(stat_base_name),
            #     oracle_env_qf_grad
            # ))
            # statistics.update(create_stats_ordered_dict(
            #     '{}_OracleQF_Grads_memory'.format(stat_base_name),
            #     oracle_memory_qf_grad
            # ))
            statistics.update(create_stats_ordered_dict(
                '{}_QfOutput'.format(stat_base_name),
                qf_output
            ))
            statistics.update(create_stats_ordered_dict(
                '{}_OracleQfOutput'.format(stat_base_name),
                oracle_qf_output
            ))
        return statistics

    def should_train_qf_extra(self, n_steps_total):
        return (
            True
            and n_steps_total % self.extra_train_period == 0
            and self.train_qf_op is not None
            and self.qf_total_loss_tolerance is not None
            and self.max_num_q_updates > 0
        )
