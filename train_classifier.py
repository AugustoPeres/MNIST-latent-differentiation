"""Trains a mnist classifier."""
from absl import app
from absl import flags

import pytorch_lightning as pl
import torch
import torchvision

from lightning_module import MNISTClassifier

FLAGS = flags.FLAGS

flags.DEFINE_integer('max_epochs', 10, 'The number of epochs.')
flags.DEFINE_bool('use_gpu', False, 'Whether to use gpu or not.')
flags.DEFINE_integer('n_layers', 5, 'The number of layers.')
flags.DEFINE_integer('batch_size', 32, 'The batch size.')


def main(_):
    train_dataset = torchvision.datasets.MNIST(
        '/tmp/mnist',
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor())
    validation_dataset = torchvision.datasets.MNIST(
        '/tmp/mnist',
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor())

    example_image = train_dataset[0][0]
    print(f'Example image shape: {example_image.shape}')
    print(f'Example image dtype: {example_image.dtype}')
    print(f'Example image min: {example_image.min()}')
    print(f'Example image max: {example_image.max()}')

    mnist_classifier = MNISTClassifier(FLAGS.n_layers)

    # Early stopping callback
    early_stopping = pl.callbacks.EarlyStopping(monitor='val_loss',
                                                patience=5,
                                                mode='min')

    trainer = pl.Trainer(accelerator='gpu' if FLAGS.use_gpu else 'cpu',
                         max_epochs=FLAGS.max_epochs,
                         callbacks=[early_stopping],
                         default_root_dir='classifier_logs')

    trainer.fit(
        mnist_classifier,
        torch.utils.data.DataLoader(train_dataset,
                                    batch_size=FLAGS.batch_size),
        torch.utils.data.DataLoader(validation_dataset,
                                    batch_size=FLAGS.batch_size))


if __name__ == '__main__':
    app.run(main)
