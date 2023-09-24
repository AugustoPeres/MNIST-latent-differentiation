from absl import app
from absl import flags

import os

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from lightning_module import (MNISTClassifier, MNISTDifferentiableGenerator,
                              EncoderDecoderWrapper)
from callbacks import StopOnLoss

FLAGS = flags.FLAGS

flags.DEFINE_integer('max_epochs', 10, 'The number of epochs.')
flags.DEFINE_integer('iterations_per_epoch', 100,
                     'Number of backpropagations per peoch.')
flags.DEFINE_float('learning_rate', 1e-3, 'The learning rate.')
flags.DEFINE_integer('class_to_generate', 0, 'The class to generate.')

flags.DEFINE_bool('use_gpu', False, 'Whether to use gpu or not.')

flags.DEFINE_string('classifier_checkpoint', None,
                    'Paths to the classifier checkpoint.')
flags.DEFINE_string('autoencoder_checkpoint', None,
                    'Path to the autoencoder checkpoint.')
flags.DEFINE_integer('latent_dim', 2, 'The latent dimension.')

flags.DEFINE_integer('seed', None, 'The random seed.')

flags.DEFINE_string('output_dir', 'generated_images', 'The output directory.')


def get_latent_space_points(encoder):
    validation_dataset = torchvision.datasets.MNIST(
        '/tmp/mnist',
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor())

    data_loader = torch.utils.data.DataLoader(validation_dataset,
                                              batch_size=1,
                                              shuffle=False)
    # Apply the encoder to the validation set
    with torch.no_grad():
        latent_space = []
        for batch in data_loader:
            x, y = batch
            x = x.view(-1, 28 * 28)
            z = encoder(x.cuda())
            latent_space.append((z.cpu().squeeze().numpy(), y.item()))
    return latent_space


def main(_):
    # Fix random seed
    if FLAGS.seed is not None:
        torch.manual_seed(FLAGS.seed)
        np.random.seed(FLAGS.seed)

    output_dir = f'{FLAGS.output_dir}_class_{FLAGS.class_to_generate}'\
                 f'_lr_{FLAGS.learning_rate}_seed_{FLAGS.seed}'

    loader = torch.utils.data.DataLoader(range(FLAGS.iterations_per_epoch),
                                         batch_size=1)
    classifier = MNISTClassifier.load_from_checkpoint(
        FLAGS.classifier_checkpoint)
    autoencoder = EncoderDecoderWrapper.load_from_checkpoint(
        FLAGS.autoencoder_checkpoint)
    encoder = autoencoder.encoder
    latent_dim = FLAGS.latent_dim

    generator = MNISTDifferentiableGenerator(classifier,
                                             autoencoder,
                                             latent_dim,
                                             FLAGS.class_to_generate,
                                             learning_rate=FLAGS.learning_rate)

    trainer = pl.Trainer(accelerator='gpu' if FLAGS.use_gpu else 'cpu',
                         max_epochs=FLAGS.max_epochs,
                         default_root_dir='generator_logs',
                         callbacks=[StopOnLoss(0.1)],
                         log_every_n_steps=1)

    initial_image = autoencoder.decoder(
        generator.latent.cuda()).detach().cpu().view((28, 28)).numpy()
    print(initial_image.shape)

    trainer.fit(generator, loader)

    image = autoencoder.decoder(generator.latent).detach().cpu().view(
        (28, 28)).numpy()
    # Plot the initial image and the final image
    plt.subplot(1, 2, 1)
    plt.imshow(initial_image, cmap='gray')
    plt.title('Initial image')
    plt.subplot(1, 2, 2)
    plt.imshow(image, cmap='gray')
    plt.title('Final image')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, 'initial_final.png'))

    latent_space_encoder = get_latent_space_points(encoder.cuda())
    # Getting all the latent vectors
    latent_vectors = generator.latents
    # Plot and save all the images
    if not os.path.exists(os.path.join(output_dir, 'latent_interpolation')):
        os.makedirs(os.path.join(output_dir, 'latent_interpolation'))
    for i, latent_vector in enumerate(latent_vectors):
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Step {i}')
        # Plot the latent space of the validation dataset
        latent_space_x = list(map(lambda x: x[0][0], latent_space_encoder))
        latent_space_y = list(map(lambda x: x[0][1], latent_space_encoder))
        y = list(map(lambda x: x[1], latent_space_encoder))

        axs[0].set_title('Latent Space')
        # Create a mapping from the 10 classes to colors
        colors = [
            'red', 'green', 'blue', 'orange', 'purple', 'yellow', 'pink',
            'black', 'brown', 'gray'
        ]
        # Elements with the same class are plotted with the same color
        im = axs[0].scatter(latent_space_x,
                            latent_space_y,
                            marker='.',
                            c=y,
                            cmap='tab10')
        # Legend with the class names
        plt.colorbar(im, ax=axs[0], ticks=range(10))
        # fix the axes to the current value
        axs[0].set_xlim(axs[0].get_xlim())
        axs[0].set_ylim(axs[0].get_ylim())

        # Plot the latent space in black on the first plot
        axs[0].scatter(latent_vector.numpy()[0],
                       latent_vector.numpy()[1],
                       marker='.',
                       c='black')

        # Plot the image corresponding to that latent space vector.
        axs[1].set_title('Image from latent')
        autoencoder.decoder.cuda()
        image = autoencoder.decoder(latent_vector.cuda())
        axs[1].imshow(image.detach().cpu().view((28, 28)).numpy(), cmap='gray')

        # Make a bar plot with the classifier probabilities.
        classifier.cuda()
        probabilities = classifier(image.view((1, 28, 28)).cuda()).view(1, 10)
        probabilities = torch.nn.functional.softmax(probabilities, dim=1)
        axs[2].bar(range(10), probabilities.detach().cpu().numpy()[0])
        axs[2].set_ylim([0, 1])
        axs[2].set_title('Classifier probabilities')
        axs[2].set_xticks(range(10))
        axs[2].set_xticklabels(range(10))

        # Save the figure.
        plt.savefig(
            os.path.join(output_dir, 'latent_interpolation',
                         f'latent_interpolation_{i:05d}.png'))
        plt.close(fig)


if __name__ == '__main__':
    app.run(main)
