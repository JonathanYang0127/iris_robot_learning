"""
Contain some self-contained modules. Maybe depend on pytorch_util.
"""
import torch
import torch.nn as nn
from railrl.torch import pytorch_util as ptu


class OuterProductLinear(nn.Module):
    def __init__(self, in_features1, in_features2, out_features, bias=True):
        super().__init__()
        self.fc = nn.Linear(
            (in_features1 + 1) * (in_features2 + 1),
            out_features,
            bias=bias,
        )

    def forward(self, in1, in2):
        out_product_flat = ptu.double_moments(in1, in2)
        return self.fc(out_product_flat)


class SelfOuterProductLinear(OuterProductLinear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, in_features, out_features, bias=bias)

    def forward(self, input):
        return super().forward(input, input)


class BatchSquareDiagonal(nn.Module):
    """
    Compute x^T diag(`diag_values`) x
    """
    def __init__(self, vector_size):
        super().__init__()
        self.vector_size = vector_size
        self.diag_mask = ptu.Variable(torch.diag(torch.ones(vector_size)),
                                      requires_grad=False)

    def forward(self, vector, diag_values):
        M = ptu.batch_diag(diag_values=diag_values, diag_mask=self.diag_mask)
        return ptu.batch_square_vector(vector=vector, M=M)


class Concat(nn.Module):
    """
    Flatten a tuple of inputs.

    Usage:
    ```
    net = nn.Sequential(
        Concat(),
        nn.Linear(3, 1),
    )

    a = Variable(torch.ones(32, 2))
    b = Variable(torch.ones(32, 1))

    result = net((a, b))
    ```
    """
    def __init__(self, *, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, inputs):
        return torch.cat(inputs, dim=self.dim)