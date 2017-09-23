import torch as t
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

        m = z
        for i, layer in enumerate(self.m):
            m = layer(m, h)
            if i != len(self.m) - 1:
                m = F.elu(m)

        s = z
        for i, layer in enumerate(self.s):
            s = layer(s, h)
            if i != len(self.s) - 1:
                s = F.elu(s)

        z = s.exp() * z + m

        log_det = s.sum(1)

        return z, log_det
