import tensorflow as tf

from rllab.core.serializable import Serializable
from railrl.core import tf_util
from railrl.core.neuralnet import NeuralNetwork
from rllab.misc.overrides import overrides


class Perceptron(NeuralNetwork):
    """A perceptron, where output = W * input + b"""
    def __init__(
            self,
            name_or_scope,
            input_tensor,
            input_size,
            output_size,
            W_name=tf_util.WEIGHT_DEFAULT_NAME,
            b_name=tf_util.BIAS_DEFAULT_NAME,
            W_initializer=None,
            b_initializer=None,
            batch_norm=False,
            batch_norm_config=None,
            **kwargs
    ):
        Serializable.quick_init(self, locals())
        super().__init__(name_or_scope, **kwargs)
        self._batch_norm = batch_norm or batch_norm_config is not None
        self._bn_stat_update_ops = []

        self.input_size = input_size
        self.output_size = output_size
        self.W_name = W_name
        self.b_name = b_name
        self.W_initializer = W_initializer
        self.b_initializer = b_initializer
        self._update_bn_ops = None
        with tf.variable_scope(name_or_scope) as variable_scope:
            output = self._create_network(input_tensor)
        if self._batch_norm:
            with tf.variable_scope(name_or_scope) as variable_scope:
                self._training_output, batch_ops = tf_util.batch_norm(
                    output,
                    is_training=True,
                    batch_norm_config=batch_norm_config,
                )
                self._update_bn_ops = batch_ops.update_pop_stats_ops
                variable_scope.reuse_variables()
                output_copy = self._create_network(input_tensor)
                self._output, _ = tf_util.batch_norm(
                    output_copy,
                    is_training=False,
                    batch_norm_config=batch_norm_config,
                )
        else:
            self._output = output
            self._training_output = output

    def _create_network(self, input_tensor):
        return tf_util.linear(
            input_tensor,
            self.input_size,
            self.output_size,
            W_name=self.W_name,
            b_name=self.b_name,
            W_initializer=self.W_initializer,
            b_initializer=self.b_initializer,
        )

    @property
    @overrides
    def training_output(self):
        return self._training_output

    @property
    def batch_norm_update_stats_op(self):
        return self._update_bn_ops
