from functools import reduce
from operator import mul
from mnist.vae import VAE

import torch as t
import torch.nn as nn
from torch.autograd import Variable


if __name__ == '__main__':

    # model = VAE()
    x = Variable(t.FloatTensor([1]))
    y = Variable(t.FloatTensor([2]))
    print(t.max(t.stack([x, y])), 0)
    # if not input > 12:
        # print('test')
    # model(input)
    # print([par.size() for par in model.parameters()])