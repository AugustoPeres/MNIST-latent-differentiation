"""This script plots the latent space of the Autoencoder"""
from absl import app
from absl import flags

import torch
import torchvision

import matplotlib.pyplot as plt

FLAGS = flags.FLAGS

flags.DEFINE_string('autoencoder_checkpoint', None,
                    'Path to the autoencoder checkpoint.')

from encoder_decoder_modules import EncoderDecoderWrapper


def main(_):
    validation_dataset = torchvision.datasets.MNIST(
        '/tmp/mnist',
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor())
    data_loader = torch.utils.data.DataLoader(validation_dataset,
                                              batch_size=1,
                                              shuffle=False)

    autoencoder = EncoderDecoderWrapper.load_from_checkpoint(
        FLAGS.autoencoder_checkpoint)
    encoder = autoencoder.encoder

    # Apply the encoder to the validation set
    with torch.no_grad():
        latent_space = []
        for batch in data_loader:
            x, y = batch
            x = x.view(-1, 28 * 28)
            z = encoder(x.cuda())
            latent_space.append((z.cpu().squeeze().numpy(), y.item()))

    # Plot the latent space in 3D where each class is a different
    # color.
    latent_space_x = list(map(lambda x: x[0][0], latent_space))
    latent_space_y = list(map(lambda x: x[0][1], latent_space))
    y = list(map(lambda x: x[1], latent_space))
    # Create a mapping from the 10 classes to colors
    colors = [
        'red', 'green', 'blue', 'orange', 'purple', 'yellow', 'pink', 'black',
        'brown', 'gray'
    ]
    # Elements with the same class are plotted with the same color

    plt.scatter(latent_space_x, latent_space_y, marker='.', c=y, cmap='tab10')
    # Legend with the class names
    plt.colorbar()
    plt.savefig('latent_space.png')


if __name__ == '__main__':
    app.run(main)
