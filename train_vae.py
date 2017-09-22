import argparse
import os
from os import listdir

import imageio
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from torch.optim import Adam
from torchvision import datasets

from mnist.vae import VAE


def make_grid(tensor, number, size):
    tensor = t.transpose(tensor, 0, 1).contiguous().view(1, number, number * size, size)
    tensor = t.transpose(tensor, 1, 2).contiguous().view(1, number * size, number * size)

    return tensor


if __name__ == "__main__":

    if not os.path.exists('prior_sampling'):
        os.mkdir('prior_sampling')

    parser = argparse.ArgumentParser(description='CDVAE')
    parser.add_argument('--num-epochs', type=int, default=3, metavar='NI',
                        help='num epochs (default: 3)')
    parser.add_argument('--batch-size', type=int, default=15, metavar='BS',
                        help='batch size (default: 15)')
    parser.add_argument('--use-cuda', type=bool, default=False, metavar='CUDA',
                        help='use cuda (default: False)')
    parser.add_argument('--learning-rate', type=float, default=0.002, metavar='LR',
                        help='learning rate (default: 0.002)')
    parser.add_argument('--save', type=str, default='trained_model', metavar='TS',
                        help='path where save trained model to (default: "trained_model")')
    args = parser.parse_args()

    dataset = datasets.MNIST(root='data/',
                             transform=transforms.Compose([
                                 transforms.ToTensor()]),
                             download=True,
                             train=True)
    dataloader = t.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    vae = VAE()
    if args.use_cuda:
        vae = vae.cuda()

    optimizer = Adam(vae.parameters(), args.learning_rate, eps=1e-6)

    likelihood_function = nn.BCEWithLogitsLoss(size_average=False)

    z = [Variable(t.randn(256, size)) for size in vae.latent_size]
    if args.use_cuda:
        z = [var.cuda() for var in z]

    repeats = vae.acc_latent_mul[0]

    for epoch in range(args.num_epochs):
        for iteration, (input, _) in enumerate(dataloader):

            input = Variable(input).view(-1, 784)
            if args.use_cuda:
                input = input.cuda()

            optimizer.zero_grad()

            out, kld = vae(input)
            input = input.unsqueeze(1).repeat(1, repeats, 1, 1, 1).view(-1, 1, 28, 28)

            likelihood = likelihood_function(out.view(-1, 1, 28, 28), input) / (args.batch_size * repeats)
            # print(likelihood, kld.mean())
            loss = likelihood + kld.mean()

            loss.backward()
            optimizer.step()

            if iteration % 10 == 0:
                print('epoch {}, iteration {}, loss {}'.format(epoch, iteration, loss.cpu().data.numpy()[0]))

                sampling = vae.sample(z).view(-1, 1, 28, 28)

                grid = make_grid(F.sigmoid(sampling).cpu().data, 16, 28)
                vutils.save_image(grid, 'prior_sampling/vae_{}.png'.format(epoch * len(dataloader) + iteration))

    samplings = [f for f in listdir('prior_sampling')]
    samplings = [imageio.imread('prior_sampling/' + path) for path in samplings for _ in range(2)]
    imageio.mimsave('prior_sampling/movie.gif', samplings)

    t.save(vae.cpu().state_dict(), args.save)
