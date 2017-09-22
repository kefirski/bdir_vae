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

    def inference(self, inference_input, repeat, type):
        """
        :param inference_input: An float tensor
        :param repeat: An int of number of repeats in order to perform monte-carlo approximation of expectations
        :return: distribution parameters
        """

        assert not self.top_most, 'Generative error. Top most block can not perform inference of posterior'
        assert type in ['posterior', 'prior']

        [bs, _] = inference_input.size()

        result = self.posterior(inference_input) if type == 'posterior' else self.prior(inference_input)
        return [var.unsqueeze(1).repeat(1, repeat, 1).view(bs * repeat, -1) if var is not None else None
                for var in result]

    def forward(self, inference_input):
        """
        :param inference_input: An float tensor with top-down input
        :return: An float tensor with out of generative function from top-down inference
        """

        assert self.top_most
        return self.out(inference_input)
