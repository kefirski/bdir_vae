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
                        nn.Linear(784, 1500),
                        nn.ELU()
                    ),
                    posterior=ParametersInference(1500, latent_size=100, h_size=100),
                    out=nn.Sequential(
                        nn.Linear(1500, 900),
                        nn.SELU()
                    )
                ),
                InferenceBlock(
                    input=nn.Sequential(
                        nn.Linear(900, 400),
                        nn.ELU(),
                        nn.Linear(400, 200),
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
                        nn.Linear(100, 100),
                        nn.SELU()
                    ),
                    prior=ParametersInference(100, latent_size=100),
                    out=GenerativeOut(nn.Sequential(
                        nn.Linear(100 + 100, 300),
                        nn.SELU(),
                        nn.Linear(300, 400),
                        nn.SELU(),
                        nn.Linear(400, 600),
                        nn.SELU(),
                        nn.Linear(600, 784),
                    )),

                ),
                GenerativeBlock(
                    out=nn.Sequential(
                        nn.Linear(30, 80),
                        nn.SELU(),
                        nn.Linear(80, 100),
                        nn.SELU()
                    )
                )
            ]
        )

        self.latent_size = [100, 30]

        assert len(self.inference) == len(self.generation) == len(self.iaf)
        self.vae_length = len(self.inference)

        self.number_of_sampling = [2, 2]

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

            parameters = [VAE.repeat(var, self.number_of_sampling[i], batch_size) for var in parameters]

            posterior_parameters.append(parameters)

        '''
        Here we perform generation in top-most layer.
        We will use posterior and prior in layers bellow this.
        '''
        [mu, std, h] = posterior_parameters[-1]

        prior = Variable(t.randn(*mu.size()))
        if cuda:
            prior.cuda()

        eps = Variable(t.randn(*mu.size()))
        if cuda:
            eps.cuda()

        posterior_gauss = eps * std + mu
        posterior, log_det = self.iaf[-1](posterior_gauss, h)

        kld = VAE.monte_carlo_divergence(z=posterior,
                                         z_gauss=posterior_gauss,
                                         log_det=log_det,
                                         posterior=[mu, std],
                                         n=self.number_of_sampling[-1])
        kld = t.max(t.stack([kld.mean(), Variable(t.FloatTensor([1]))]), 0)[0]

        posterior = VAE.unrepeat(posterior, self.number_of_sampling[-1], batch_size)
        prior = VAE.unrepeat(prior, self.number_of_sampling[-1], batch_size)

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
            [top_down_mu, top_down_std] = [VAE.repeat(var, self.number_of_sampling[i], batch_size)
                                           for var in [top_down_mu, top_down_std]]
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
            [prior_mu, prior_std] = [VAE.repeat(var, self.number_of_sampling[i], batch_size)
                                     for var in [prior_mu, prior_std]]

            _kld = VAE.monte_carlo_divergence(z=posterior,
                                              z_gauss=posterior_gauss,
                                              log_det=log_det,
                                              posterior=[posterior_mu, posterior_std],
                                              prior=[prior_mu, prior_std],
                                              n=self.number_of_sampling[i])
            _kld = t.max(t.stack([_kld.mean(), Variable(t.FloatTensor([1]))]), 0)[0]
            kld += _kld

            posterior_determenistic = VAE.repeat(posterior_determenistic, self.number_of_sampling[i], batch_size)
            posterior = self.generation[i].out(posterior, posterior_determenistic)

            posterior = VAE.unrepeat(posterior, self.number_of_sampling[i], batch_size)
            if i != 0:
                '''
                Since there no level below bottom-most, 
                there no reason to pass prior through out operation
                '''

                eps = Variable(t.randn(*prior_mu.size()))
                if cuda:
                    eps.cuda()

                prior = eps * prior_std + prior_mu

                prior_determenistic = VAE.repeat(prior_determenistic, self.number_of_sampling[i], batch_size)
                prior = self.generation[i].out(prior, prior_determenistic)
                prior = VAE.unrepeat(prior, self.number_of_sampling[i], batch_size)

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
        if kwargs['z'].is_cuda:
            for var in kwargs['prior']:
                var.cuda()
        log_p_z = VAE.log_gauss(kwargs['z'], kwargs['prior'])

        return log_p_z_x - log_p_z

    @staticmethod
    def log_gauss(z, params):
        [mu, std] = params
        return - 0.5 * (t.pow(z - mu, 2) * t.pow(std + 1e-8, -2) + 2 * t.log(std + 1e-8) + log(2 * pi)).sum(1)

    @staticmethod
    def repeat(input, n, batch_size):
        return input.unsqueeze(1).repeat(1, n, 1).view(batch_size * n, -1)

    @staticmethod
    def unrepeat(input, n, batch_size):
        return input.view(batch_size, n, -1)[:, 0, :]