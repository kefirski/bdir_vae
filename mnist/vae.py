from math import log, pi

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
                        nn.utils.weight_norm(nn.Linear(784, 1500)),
                        nn.ELU()
                    ),
                    posterior=ParametersInference(1500, latent_size=100, h_size=100),
                    out=nn.Sequential(
                        nn.utils.weight_norm(nn.Linear(1500, 900)),
                        nn.SELU()
                    )
                ),
                InferenceBlock(
                    input=nn.Sequential(
                        nn.utils.weight_norm(nn.Linear(900, 400)),
                        nn.ELU(),
                        nn.utils.weight_norm(nn.Linear(400, 200)),
                        nn.ELU(),
                    ),
                    posterior=ParametersInference(200, latent_size=30, h_size=50)
                )
            ]
        )

        self.iaf = nn.ModuleList(
            [
                IAF(latent_size=100, h_size=100),
                IAF(latent_size=30, h_size=50)
            ]
        )

        self.generation = nn.ModuleList(
            [
                GenerativeBlock(
                    posterior=ParametersInference(100, latent_size=100),
                    input=nn.Sequential(
                        nn.utils.weight_norm(nn.Linear(100, 100)),
                        nn.SELU()
                    ),
                    prior=ParametersInference(100, latent_size=100),
                    out=GenerativeOut(nn.Sequential(
                        nn.utils.weight_norm(nn.Linear(100 + 100, 300)),
                        nn.SELU(),
                        nn.utils.weight_norm(nn.Linear(300, 400)),
                        nn.SELU(),
                        nn.utils.weight_norm(nn.Linear(400, 600)),
                        nn.SELU(),
                        nn.utils.weight_norm(nn.Linear(600, 784)),
                    )),

                ),
                GenerativeBlock(
                    out=nn.Sequential(
                        nn.utils.weight_norm(nn.Linear(30, 80)),
                        nn.SELU(),
                        nn.utils.weight_norm(nn.Linear(80, 100)),
                        nn.SELU()
                    )
                )
            ]
        )

        self.latent_size = [100, 30]

        assert len(self.inference) == len(self.generation) == len(self.iaf)
        self.vae_length = len(self.inference)

    def forward(self, input):
        """
        :param input: An float tensor with shape of [batch_size, 784]
        :return: An float tensor with shape of [batch_size, 784]
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

            posterior_parameters.append(parameters)

        '''
        Here we perform generation in top-most layer.
        We will use posterior and prior in layers bellow this.
        '''
        [mu, std, h] = posterior_parameters[-1]

        prior = Variable(t.randn(*mu.size()))
        eps = Variable(t.randn(*mu.size()))

        if cuda:
            prior, eps = prior.cuda(), eps.cuda()

        posterior_gauss = eps * std + mu
        posterior, log_det = self.iaf[-1](posterior_gauss, h)

        kld = VAE.monte_carlo_divergence(z=posterior,
                                         z_gauss=posterior_gauss,
                                         log_det=log_det,
                                         posterior=[mu, std])

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
            [top_down_mu, top_down_std, _] = self.generation[i].inference(posterior, 'posterior')
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
            prior_mu, prior_std, _ = self.generation[i].inference(prior_determenistic, 'prior')

            kld += VAE.monte_carlo_divergence(z=posterior,
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

                eps = Variable(t.randn(*prior_mu.size()))
                if cuda:
                    eps.cuda()

                prior = eps * prior_std + prior_mu

                prior = self.generation[i].out(prior, prior_determenistic)

        return posterior, kld

    def sample(self, z):
        """
        :param z: An array of variables from normal distribution each with shape of [batch_size, latent_size[i]]
        :return: Sample from generative model with shape of [batch_size, 784]
        """

        top_variable = z[-1]

        out = self.generation[-1].out(top_variable)

        for i in range(self.vae_length - 2, -1, -1):
            determenistic = self.generation[i].input(out)

            [mu, std, _] = self.generation[i].prior(determenistic)
            prior = z[i] * std + mu
            out = self.generation[i].out(prior, determenistic)

        return out

    @staticmethod
    def monte_carlo_divergence(**kwargs):

        log_p_z_x = VAE.log_gauss(kwargs['z_gauss'], kwargs['posterior']) - kwargs['log_det']

        if kwargs.get('prior') is None:
            kwargs['prior'] = [Variable(t.zeros(*kwargs['z'].size())),
                               Variable(t.ones(*kwargs['z'].size()))]

        one = Variable(t.FloatTensor([1]))

        if kwargs['z'].is_cuda:
            one = one.cuda()
            for var in kwargs['prior']:
                var.cuda()
        log_p_z = VAE.log_gauss(kwargs['z'], kwargs['prior'])

        result = log_p_z_x - log_p_z
        return t.max(t.stack([result.mean(), one]), 0)[0]

    @staticmethod
    def log_gauss(z, params):
        [mu, std] = params
        return - 0.5 * (t.pow(z - mu, 2) * t.pow(std + 1e-8, -2) + 2 * t.log(std + 1e-8) + log(2 * pi)).sum(1)
