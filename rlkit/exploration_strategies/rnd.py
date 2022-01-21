import rlkit.torch.pytorch_util as ptu
from rlkit.torch.networks.cnn import CNN
from torch import nn
import torch

class RNDModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        
        self.f = CNN(**kwargs, hidden_init=nn.init.uniform_)
        self.f_hat = CNN(**kwargs, hidden_init=nn.init.normal_)

        self.f.to(ptu.device)
        self.f_hat.to(ptu.device)

    def forward(self, ob_no):
        targets = self.f(ob_no).detach()
        predictions = self.f_hat(ob_no)
        return torch.norm(predictions - targets, dim=1)

    def forward_np(self, ob_no):
        ob_no = ptu.from_numpy(ob_no)
        error = self(ob_no)
        return ptu.to_numpy(error)
