from functools import reduce
from operator import mul
from mnist.vae import VAE

import torch as t
import torch.nn as nn
from torch.autograd import Variable


if __name__ == '__main__':
    model = VAE()

    input = Variable(t.randn(5, 784))

    model(input)