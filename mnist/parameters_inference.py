import torch.nn as nn


class ParametersInference(nn.Module):
    def __init__(self, input_size, latent_size, h_size=None):
        super(ParametersInference, self).__init__()

        self.mu = nn.Linear(input_size, latent_size)
        self.std = nn.Linear(input_size, latent_size)

        self.h = nn.Linear(input_size, h_size) if h_size is not None else None

    def forward(self, input):
        mu = self.mu(input)
        std = self.std(input).exp()
        h = self.h(input) if self.h is not None else None

        return mu, std, h
