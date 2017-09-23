import torch.nn as nn
import torch.nn.functional as F

from IAF.autoregressive_linear import AutoregressiveLinear
from IAF.highway import Highway


class IAF(nn.Module):
    def __init__(self, latent_size, h_size):
        super(IAF, self).__init__()

        self.z_size = latent_size
        self.h_size = h_size

        self.h = nn.Sequential(
            Highway(self.h_size, 3, nn.ELU()),
            nn.Linear(self.h_size, self.z_size * self.z_size)
        )

        self.m = nn.ModuleList([
            AutoregressiveLinear(self.z_size),
            AutoregressiveLinear(self.z_size)
        ])
        self.s = nn.ModuleList([
            AutoregressiveLinear(self.z_size),
            AutoregressiveLinear(self.z_size)
        ])

    def forward(self, z, h):
        """
        :param z: An float tensor with shape of [batch_size, z_size]
        :param h: An float tensor with shape of [batch_size, h_size]
        :return: An float tensor with shape of [batch_size, z_size] and log det value of jacobian of the IAF mapping
        """

        h = self.h(h)
        h = h.view(-1, self.z_size, self.z_size)

        m = IAF.unroll_autogressive_network(z, self.m, h)
        s = IAF.unroll_autogressive_network(z, self.s, h)

        z = s.exp() * z + m

        log_det = s.sum(1)

        return z, log_det

    @staticmethod
    def unroll_autogressive_network(input, layers, h):

        for i, layer in enumerate(layers):
            input = layer(input, h)
            if i != len(layers) - 1:
                input = F.elu(input)

        return input
