from math import log, pi
from functools import reduce
from operator import mul

import torch as t
import torch.nn as nn
from torch.autograd import Variable

from mnist.parameters_inference import ParametersInference
from mnist.generative_out import GenerativeOut
from IAF.IAF import IAF
from blocks.inference_block import InferenceBlock
from blocks.generative_block import GenerativeBlock


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.inference = nn.ModuleList(
            [
                InferenceBlock(
                    input=nn.Sequential(
                        nn.Linear(784, 1500),
                        nn.ELU()
                    ),
                    posterior=ParametersInference(1500, latent_size=140, h_size=400),
                    out=nn.Sequential(
                        nn.Linear(1500, 1200),
                        nn.ELU(),
                        nn.Linear(1200, 900),
                        nn.ELU()
                    )
                ),

                InferenceBlock(
                    input=nn.Sequential(
                        nn.Linear(900, 500),
                        nn.ELU(),
                        nn.Linear(500, 300),
                        nn.ELU(),
                        nn.Linear(300, 100),
                        nn.ELU()
                    ),
                    posterior=ParametersInference(100, latent_size=30, h_size=50)
                )
            ]
        )

        self.iaf = nn.ModuleList(
            [
                IAF(latent_size=140, h_size=400),
                IAF(latent_size=30, h_size=50)
            ]
        )

        self.generation = nn.ModuleList(
            [
                GenerativeBlock(
                    out=GenerativeOut(nn.Sequential(
                        nn.Linear(140 + 120, 300),
                        nn.ELU(),
                        nn.Linear(300, 400),
                        nn.ELU(),
                        nn.Linear(400, 600),
                        nn.ELU(),
                        nn.Linear(600, 784),
                    )),
                    posterior=ParametersInference(90, latent_size=140),
                    input=nn.Sequential(
                        nn.Linear(90, 120),
                        nn.ELU()
                    ),
                    prior=ParametersInference(120, latent_size=140)

                ),

                GenerativeBlock(
                    out=nn.Sequential(
                        nn.Linear(30, 90),
                        nn.ELU()
                    )
                )
            ]
        )

        self.latent_size = [140, 30]

        '''
        In order to approximate kl-divergence 
        it is necessary to sample latent_mul[i] number of latent variables at i-th layer of network
        '''
        self.latent_mul = [5, 5]
        self.acc_latent_mul = [reduce(mul, self.latent_mul[::-1][0:i + 1], 1)
                               for i, _ in enumerate(self.latent_mul)][::-1]

        assert len(self.inference) == len(self.generation) == len(self.iaf)
        self.vae_length = len(self.inference)

    def forward(self, input):
        """
        :param input: An float tensor with shape of [batch_size, 784]
        :return: An float tensor with shape of [batch_size * acc_latent_mul[0], 784]
                     with logits of margin likelihood expectation
        """

        [batch_size, _] = input.size()

        latent_parameters = []

        for i in range(self.vae_length):
            if i < self.vae_length - 1:
                input, parameters = self.inference[i](input)
            else:
                parameters = self.inference[i](input)
            acc_size = self.acc_latent_mul[i]
            parameters = [var.unsqueeze(1).repeat(1, acc_size, 1).view(batch_size * acc_size, -1) for var in parameters]
            latent_parameters.append(parameters)

        return None
