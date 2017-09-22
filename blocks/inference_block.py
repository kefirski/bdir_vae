import torch.nn as nn


class InferenceBlock(nn.Module):
    def __init__(self, **kwargs):
        super(InferenceBlock, self).__init__()

        '''
        Computation graph of inference block is identical to described in
        "Improved Variational Inference with Inverse Autoregressive Flow" paper.

        Firstly input goes through input operation in order to get some hidden state,
        that is used in order to get features of posterior distribution.

        Then this hidden state are passed to out operation.
        '''

        for name, value in kwargs.items():
            setattr(self, name, value)

        self.top_most = kwargs.get('out') is None

    def forward(self, input):
        """
        :param input: An float tensor with shape appropriate to first_op
        :return: result of out operation (None if top_most) and posterior features
        """

        hidden_state = self.input(input)
        posterior_parameters = self.posterior(hidden_state)

        if self.top_most:
            return posterior_parameters

        result = self.out(hidden_state) if not self.top_most else None
        return result, posterior_parameters
