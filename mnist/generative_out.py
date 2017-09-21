import torch as t
import torch.nn as nn


class GenerativeOut(nn.Module):
    def __init__(self, fc):
        super(GenerativeOut, self).__init__()

        self.fc = fc

    def forward(self, latent_input, determenistic_input):
        [lat_bs, _] = latent_input.size()
        [det_bs, det_size] = determenistic_input.size()

        '''
        Since in order to approximate kl-divergence with monte-carlo estimator,
        it is necessary to repeat latent variable parameters,
        we have to repeat determenistic input too.
        '''
        if lat_bs != det_bs:
            acc = int(lat_bs / det_bs)
            determenistic_input = determenistic_input.unsqueeze(1).repeat(1, acc, 1).view(-1, det_size)

        input = t.cat([latent_input, determenistic_input], 1)

        return self.fc(input)
