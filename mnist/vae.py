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
        cuda = input.is_cuda

        posterior_parameters = []

        for i in range(self.vae_length):
            if i < self.vae_length - 1:
                input, parameters = self.inference[i](input)
            else:
                parameters = self.inference[i](input)
            acc_size = self.acc_latent_mul[i]
            parameters = [var.unsqueeze(1).repeat(1, acc_size, 1).view(batch_size * acc_size, -1) for var in parameters]
            posterior_parameters.append(parameters)

        eps = Variable(t.randn(batch_size * self.acc_latent_mul[-1], self.latent_size[-1]))
        if cuda:
            eps.cuda()

        z_gauss = eps * posterior_parameters[-1][1] + posterior_parameters[-1][0]
        z, log_det = self.iaf[-1](z_gauss, posterior_parameters[-1][2])

        kld = VAE.monte_carlo_divergence(n=self.acc_latent_mul[-1],
                                         z=z,
                                         z_gauss=z_gauss,
                                         log_det=log_det,
                                         posterior=posterior_parameters[-1][:2])
        print(kld)

    @staticmethod
    def monte_carlo_divergence(**kwargs):
        """
        :param n: number of samples in random variables
        :param z: z from posterior distribution
        :param z_gauss: z from diagonal gaussian distribution
        :param log_det: log det of iaf mapping
        :param posterior = [mu_1, std_1]: parameters of posterior diagonal gaussian
        :param prior = [mu_2, std_2] [Optional]: parameters of prior diagonal gaussian
        :return: kl-divergence approximation
        """
        [batch_size, latent_size] = kwargs['posterior'][0].size()

        log_p_z_x = VAE.log_gauss(kwargs['z_gauss'], kwargs['posterior']) - kwargs['log_det']

        if kwargs.get('prior') is None:
            kwargs['prior'] = [Variable(t.zeros(batch_size, latent_size)),
                               Variable(t.ones(batch_size, latent_size))]
        if kwargs['z'].is_cuda:
            for var in kwargs['prior']:
                var.cuda()
        log_p_z = VAE.log_gauss(kwargs['z'], kwargs['prior'])

        result = log_p_z_x - log_p_z
        return result.view(-1, kwargs['n']).mean(1)

    @staticmethod
    def log_gauss(z, params):
        [mu, std] = params
        return - 0.5 * (t.pow(z - mu, 2) * t.pow(std + 1e-8, -2) + 2 * t.log(std + 1e-8) + log(2 * pi)).sum(1)