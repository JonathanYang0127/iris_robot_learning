import abc
import tensorflow as tf

from railrl.core import tf_util
from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides
from sandbox.rocky.tf.core.parameterized import Parameterized

ALLOWABLE_TAGS = ['regularizable']


def negate(function):
    return lambda x: not function(x)


class NeuralNetwork(Parameterized, Serializable):
    """
    Any neural network.
    """

    def __init__(
            self,
            name_or_scope,
            batch_norm=False,
            batch_norm_config=None,
            reuse=False,
            **kwargs
    ):
        super().__init__(**kwargs)
        Serializable.quick_init(self, locals())
        if type(name_or_scope) is str:
            self.scope_name = name_or_scope
        else:
            self.scope_name = name_or_scope.original_name_scope
        self._batch_norm = batch_norm or batch_norm_config is not None
        self._batch_norm_config = batch_norm_config
        self._reuse = reuse
        self._bn_stat_update_ops = []
        self._output = None
        self._sess = None
        self._training_output = None
        self._output = None
        self._variable_scope = None

    def _create_network(self, **inputs):
        """
        This method should be called by subclasses after the super call is made.

        :param inputs: named Tensors
        :return: None
        """
        with tf.variable_scope(
                self.scope_name, reuse=self._reuse
        ) as self._variable_scope:
            if self._batch_norm:
                self._training_output, self._output = (
                    self._create_network_internal(**inputs)
                )
            else:
                self._output = self._create_network_internal(**inputs)
                self._training_output = self._output

    @property
    def sess(self):
        if self._sess is None:
            self._sess = tf.get_default_session()
        return self._sess

    @sess.setter
    def sess(self, value):
        self._sess = value

    @property
    def output(self):
        """
        :return: Tensor/placeholder/op. Output of this network.
        """
        return self._output

    @property
    def training_output(self):
        """
        :return: Tensor/placeholder/op. Training output of this network.
        """
        return self.output

    def _process_layer(self, previous_layer):
        """
        This should be done called between every layer, i.e.

        a = self.process_layer(linear(x))
        b = self.process_layer(linear(relu(a)))

        If batch norm is disabled, this just returns `previous_layer`
        immediately.

        If batch norm is enabled, this returns a tuple (a, b) where
            a = layer after batch norm to be used when training
            b = layer after batch norm to be used when evaluating

        :param previous_layer: Either the input layer or the output to a
        previous call to `_process_layer`
        :return: If batch norm is disabled, this returns previous_layer.
        Otherwise, it returns a tuple (batch norm'd layer for training,
        batch norm'd layer for eval)
        """
        if not self._batch_norm:
            return previous_layer

        if not isinstance(previous_layer, tuple):  # This is the input
            # TODO(vpong): Enforce that this is only called when processing
            # the input. Probably use a context manager and force subclasses
            # to do something like `with self.process_input`: ...
            # or do something more elegant
            previous_training_layer = previous_layer
            previous_eval_layer = previous_layer
        else:
            assert len(previous_layer) == 2
            previous_training_layer, previous_eval_layer = previous_layer
        training_output, batch_ops = tf_util.batch_norm(
            previous_training_layer,
            is_training=True,
            batch_norm_config=self._batch_norm_config,
        )
        self._bn_stat_update_ops += batch_ops.update_pop_stats_ops
        with tf.variable_scope(self.variable_scope, reuse=True):
            eval_output, _ = tf_util.batch_norm(
                previous_eval_layer,
                is_training=False,
                batch_norm_config=self._batch_norm_config,
            )
        return training_output, eval_output

    @overrides
    def get_params_internal(self, **tags):
        for key in tags.keys():
            if key not in ALLOWABLE_TAGS:
                raise KeyError(
                    "Tag not allowed: {0}. Allowable tags: {1}".format(
                        key,
                        ALLOWABLE_TAGS))
        # TODO(vpong): This is a big hack! Fix this
        filters = []
        if 'regularizable' in tags:
            regularizable_vars = tf_util.get_regularizable_variables(
                self.scope_name)
            if tags['regularizable']:
                reg_filter = lambda v: v in regularizable_vars
            else:
                reg_filter = lambda v: v not in regularizable_vars
            filters.append(reg_filter)

        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                      self.scope_name)
        return list(filter(lambda v: all(f(v) for f in filters), variables))

    def get_copy(self, **kwargs):
        return Serializable.clone(
            self,
            **kwargs
        )

    def get_weight_tied_copy(self, **inputs):
        """
        Return a weight-tied copy of the network. Replace the action or
        observation to the network for the returned network.

        :param inputs: Dictionary, of the form
        {
            'input_x': self.input_x,
            'input_y': self.input_y,
        }

        :return: StateNetwork copy with weights tied to this StateNetwork.
        """
        assert len(inputs) > 0
        for input_name, input_value in self._input_name_to_values.items():
            if input_name not in inputs:
                inputs[input_name] = input_value
        return self.get_copy(
            name_or_scope=self.variable_scope,
            reuse=True,
            **inputs
        )

    def setup_serialization(self, init_locals):
        # TODO(vpong): fix this
        # Serializable.quick_init_for_clone(self, init_locals)
        # init_locals_copy = dict(init_locals.items())
        # if 'kwargs' in init_locals:
        #     init_locals_copy['kwargs'].pop('action_input', None)
        #     init_locals_copy['kwargs'].pop('observation_input', None)
        # Serializable.quick_init(self, init_locals_copy)
        Serializable.quick_init(self, init_locals)

    @property
    def batch_norm_update_stats_op(self):
        return self._bn_stat_update_ops

    @abc.abstractmethod
    def _create_network_internal(self, **inputs):
        """
        This function should construct the network. Between each layer,
        it should call `self.add_layer()` like so:
        :param inputs: Tensor inputs to the network by name
        :return: Tensor, output of network
        """
        pass

    @property
    def variable_scope(self):
        return self._variable_scope

    @property
    @abc.abstractmethod
    def _input_name_to_values(self):
        """
        Return a dictionary describing what inputs are and their current values.

        :return: Return a dictionary of the form
        {
            'input_x': self.input_x,
            'input_y': self.input_y,
        }
        This will be the input to get_weight_tied_copy
        """
        pass
