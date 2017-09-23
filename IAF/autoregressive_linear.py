import math

import torch as t
import torch.nn as nn
from torch.nn.parameter import Parameter


class AutoregressiveLinear(nn.Module):
    def __init__(self, out_size, bias=True):
        super(AutoregressiveLinear, self).__init__()

        self.out_size = out_size

        if bias:
            self.bias = Parameter(t.Tensor(self.out_size))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_size)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, weight):

        # Unforchantly there is no way to perform batch matrix-vector product
        if self.bias is not None:
            output = [t.addmv(self.bias, weight[i], input[i]) for i, var in enumerate(input)]

            return t.stack(output)

        output = [t.mv(weight[i], input[i]) for i, var in enumerate(input)]

        return t.stack(output)