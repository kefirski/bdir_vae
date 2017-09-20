import torch.nn as nn


class GenerativeBlock(nn.Module):
    def __init__(self, **kwargs):
        super(GenerativeBlock, self).__init__()

        '''
        Computation graph of generative block is identical to described in
        "Improved Variational Inference with Inverse Autoregressive Flow" paper.
        
        It is quite hard to describe it's flow, 
        as hard to make input go through one function.
        '''

        for name, value in kwargs.items():
            setattr(self, name, value)

        self.top_most = kwargs.get('input') is None

    def inference(self, inference_input):
        """
        :param inference_input: An float tensor
        :return: Posterior parameters
        """

        assert not self.top_most, 'Generative error. Top most block can not perform inference of posterior'
        return self.posterior(inference_input)

    def forward(self, inference_input, prior_input, z):
        """
        :param inference_input: An float tensor with top-down input
        :param prior_input: An float tensor with input, sampled from prior distribution
        :param z: An float tensor with latent variable, sampled from posterior distribution
        :return: An float tensor with out of generative function from top-down inference
                     and sampled variable from prior distribution
        """

        '''
        This function is necessary to perfrom top-down inference.
        
        Given inference and prior input it firsly gets determenistic features.
        After this, determenistic features of prior variable are used 
        in order to get parameters of prior distribution.
        
        Inference determenistic features are combined with posterior latent variable 
        and goes through out operation.
        '''

        if self.top_most:
            return self.out(inference_input)

        else:
            inference_input = self.input(inference_input)
            prior_input = self.input(prior_input)

            prior = self.prior(prior_input)

            out = self.out(inference_input, z)

            return out, prior
