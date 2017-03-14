import tensorflow as tf

from railrl.policies.memory.rnn_cell_policy import RnnCellPolicy


class _LstmLinearCell(tf.nn.rnn_cell.LSTMCell):
    """
    LSTM cell with a linear unit + softmax before the output.
    """
    def __init__(
            self,
            num_units,
            output_dim,
            **kwargs
    ):
        super().__init__(num_units, **kwargs)
        self._output_dim = output_dim

    def __call__(self, inputs, state, scope=None):
        split_state = tf.split(1, 2, state)
        lstm_output, lstm_state = super().__call__(inputs, split_state,
                                                   scope=scope)
        flat_state = tf.concat(1, lstm_state)

        with tf.variable_scope('softmax'):
            W = tf.get_variable('W', [self._num_units, self._output_dim])
            b = tf.get_variable('b', [self._output_dim],
                                initializer=tf.constant_initializer(0.0))

        env_action_logit = tf.matmul(lstm_output, W) + b
        return tf.nn.softmax(env_action_logit), flat_state


class LstmMemoryPolicy(RnnCellPolicy):
    """
    write = affine function of environment observation and memory
    logits = affine function of environment observation, memory, and write
    action = softmax(logits)
    """

    def __init__(
            self,
            name_or_scope,
            action_dim,
            memory_dim,
            **kwargs
    ):
        assert memory_dim % 2 == 0
        self.setup_serialization(locals())
        self._memory_dim = memory_dim
        self._action_dim = action_dim
        self._rnn_cell = None
        self._rnn_cell_scope = None
        self._num_lstm_units = self._memory_dim / 2
        super().__init__(name_or_scope=name_or_scope, **kwargs)

    def _create_network_internal(self, observation_input=None):
        assert observation_input is not None
        env_obs, memory_obs = observation_input
        self._rnn_cell = _LstmLinearCell(
            self._num_lstm_units,
            self._action_dim,
        )
        with tf.variable_scope("lstm") as self._rnn_cell_scope:
            cell_output = self._rnn_cell(env_obs, memory_obs)
        return cell_output

    @property
    def rnn_cell(self):
        return self._rnn_cell

    def create_init_state_placeholder(self):
        return tf.placeholder(
            tf.float32,
            [None, self._num_lstm_units * 2],
            name='lstm_init_state',
        )

    @property
    def rnn_cell_scope(self):
        return self._rnn_cell_scope

