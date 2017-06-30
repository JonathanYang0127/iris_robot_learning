import abc

from torch import nn as nn

from railrl.torch import pytorch_util as ptu
from rllab.core.serializable import Serializable


class PyTorchModule(nn.Module, Serializable, metaclass=abc.ABCMeta):

    def get_param_values(self):
        return [ptu.get_numpy(param) for param in self.parameters()]

    def set_param_values(self, param_values):
        for param, value in zip(self.parameters(), param_values):
            param.data = ptu.from_numpy(value)

    def copy(self):
        copy = Serializable.clone(self)
        ptu.copy_model_params_from_to(self, copy)
        return copy

    def save_init_params(self, locals):
        """
        Should call this FIRST THING in the __init__ method if you ever want
        to serialize or clone this network.
        
        Usage:
        ```
        def __init__(self, ...):
            self.init_serialization(locals())
            ...
        ```
        :param locals: 
        :return: 
        """
        Serializable.quick_init(self, locals)

    def __getstate__(self):
        d = Serializable.__getstate__(self)
        d["params"] = self.get_param_values()
        return d

    def __setstate__(self, d):
        Serializable.__setstate__(self, d)
        self.set_param_values(d["params"])