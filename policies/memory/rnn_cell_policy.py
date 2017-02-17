import abc

from railrl.policies.memory.memory_policy import MemoryPolicy


class RnnCellPolicy(MemoryPolicy, metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def rnn_cell(self):
        """
        Return a TensorFlow RNNCell.
        :return:
        """
        pass

    @property
    @abc.abstractmethod
    def rnn_cell_scope(self):
        """
        Return scope under which self.rnn_cell is made Tensorflow RNNCell.
        :return:
        """
        pass
