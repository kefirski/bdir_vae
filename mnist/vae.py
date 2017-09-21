from functools import reduce
from math import log, pi
from operator import mul

import torch as t
import torch.nn as nn
from torch.autograd import Variable

from IAF.IAF import IAF
from blocks.generative_block import GenerativeBlock
from blocks.inference_block import InferenceBlock
from mnist.generative_out import GenerativeOut
from mnist.parameters_inference import ParametersInference


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

        '''
        Here we perform top-down inference.
        parameters array is filled with posterior parameters [mu, std, h]
        '''
        for i in range(self.vae_length):
            if i < self.vae_length - 1:
                input, parameters = self.inference[i](input)
            else:
                parameters = self.inference[i](input)
            acc_size = self.acc_latent_mul[i]
            parameters = [var.unsqueeze(1).repeat(1, acc_size, 1).view(batch_size * acc_size, -1)
                          for var in parameters]
            posterior_parameters.append(parameters)

        '''
        Here we perform generation in top-most layer.
        We will use posterior and prior in layers bellow this.
        '''
        prior = Variable(t.randn(*posterior_parameters[-1][0].size()))
        if cuda:
            prior.cuda()

        posterior_gauss = prior * posterior_parameters[-1][1] + posterior_parameters[-1][0]
        posterior, log_det = self.iaf[-1](posterior_gauss, posterior_parameters[-1][2])

        kld = VAE.monte_carlo_divergence(n=self.acc_latent_mul[-1],
                                         z=posterior,
                                         z_gauss=posterior_gauss,
                                         log_det=log_det,
                                         posterior=posterior_parameters[-1][:2])

        posterior = self.generation[-1](posterior)
        prior = self.generation[-1](prior)

        for i in range(self.vae_length - 2, -1, -1):

            '''
            Iteration over not top-most generative layers.
            Firstly we pass input through inputs operation in order to get determenistic features
            '''

            posterior_determenistic = self.generation[i].input(posterior)
            prior_determenistic = self.generation[i].input(prior)

            '''
            Then posterior input goes through inference function in order to get top-down features.
            Parameters of posterior are combined together and new latent variable is sampled
            '''

            [top_down_mu, top_down_std, _] = self.generation[i].inference(posterior, self.latent_mul[i], 'posterior')
            [bottom_up_mu, bottom_up_std, h] = posterior_parameters[i]

            posterior_mu = top_down_mu + bottom_up_mu
            posterior_std = top_down_std + bottom_up_std

            eps = Variable(t.randn(*posterior_mu.size()))
            if cuda:
                eps.cuda()

            posterior_gauss = eps * posterior_std + posterior_mu
            posterior, log_det = self.iaf[i](posterior_gauss, h)

            '''
            Prior parameters are obtained from prior operation,
            then new prior variable is sampled
            '''
            prior_mu, prior_std, _ = self.generation[i].inference(prior_determenistic, self.latent_mul[i], 'prior')
            eps = Variable(t.randn(*prior_mu.size()))
            if cuda:
                eps.cuda()

            prior = eps * prior_std + prior_mu

            kld += VAE.monte_carlo_divergence(n=self.acc_latent_mul[i],
                                              z=posterior,
                                              z_gauss=posterior_gauss,
                                              log_det=log_det,
                                              posterior=[posterior_mu, posterior_std],
                                              prior=[prior_mu, prior_std])

            posterior = self.generation[i].out(posterior, posterior_determenistic)
            if i != 0:
                '''
                Since there no level below bottom-most, 
                there no reason to pass prior through out operation
                '''
                prior = self.generation[i].out(prior, prior_determenistic)

        return posterior

    @staticmethod
    def monte_carlo_divergence(**kwargs):

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
