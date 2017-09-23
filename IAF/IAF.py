import torch as t
import torch.nn as nn

from IAF.autoregressive_linear import AutoregressiveLinear
from IAF.highway import Highway


class IAF(nn.Module):
    def __init__(self, latent_size, h_size):
        super(IAF, self).__init__()

        self.z_size = latent_size
        self.h_size = h_size

        self.h = Highway(self.h_size, 3, nn.ELU())

        self.m = nn.Sequential(
            AutoregressiveLinear(self.z_size + self.h_size, 2 * self.z_size, temperature=0),
            nn.ELU(),
            AutoregressiveLinear(2 * self.z_size, self.z_size, temperature=0)

        )
        self.s = nn.Sequential(
            AutoregressiveLinear(self.z_size + self.h_size, 2 * self.z_size, temperature=0),
            nn.ELU(),
            AutoregressiveLinear(2 * self.z_size, self.z_size, temperature=0)

        )

    def forward(self, z, h):
        """
        :param z: An float tensor with shape of [batch_size, z_size]
        :param h: An float tensor with shape of [batch_size, h_size]
        :return: An float tensor with shape of [batch_size, z_size] and log det value of jacobian of the IAF mapping
        """

        h = self.h(h)

        input = t.cat([z, h], 1)

        m = self.m(input)
        s = self.s(input)

        z = s.exp() * z + m

        log_det = s.sum(1)

        return z, log_det