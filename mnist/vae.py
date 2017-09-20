from math import log, pi

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
                    posterior=ParametersInference(1500, latent_size=140, h_size=140),
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
                IAF(latent_size=140, h_size=140),
                IAF(latent_size=30, h_size=50)
            ]
        )

        self.generation = nn.ModuleList(
            [
                GenerativeBlock(
                    out=nn.Sequential(
                        nn.Linear(30, 90),
                        nn.ELU()
                    )
                ),

                GenerativeBlock(
                    posterior=ParametersInference(90, latent_size=140),
                    input=nn.Sequential(
                        nn.Linear(90, 120),
                        nn.ELU()
                    ),
                    prior=ParametersInference(120, latent_size=140),
                    out=GenerativeOut(nn.Sequential(
                        nn.Linear(140 + 120, 300),
                        nn.ELU(),
                        nn.Linear(300, 400),
                        nn.ELU(),
                        nn.Linear(400, 600),
                        nn.ELU(),
                        nn.Linear(600, 784),
                        nn.ELU(),
                    ))
                )
            ]
        )

    def forward(self):
        pass
