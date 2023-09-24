from absl import app
from absl import flags

import pytorch_lightning as pl
import torch
import torchvision

import matplotlib.pyplot as plt

from  lightning_module EncoderDecoderWrapper

FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 32, 'Batch size.')
flags.DEFINE_integer('max_epochs', 10, 'Number of epochs.')
flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate.')
flags.DEFINE_integer('latent_dim', 2, 'Dimensionality of the latent space.')

flags.DEFINE_bool('use_gpu', True, 'Whether to use GPU or not.')


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

    model = EncoderDecoderWrapper(FLAGS.latent_dim, FLAGS.learning_rate)

    # Early stopping callback
    early_stopping = pl.callbacks.EarlyStopping(monitor='val_loss',
                                                patience=5,
                                                mode='min')

    trainer = pl.Trainer(accelerator='gpu' if FLAGS.use_gpu else 'cpu',
                         max_epochs=FLAGS.max_epochs,
                         callbacks=[early_stopping],
                         default_root_dir='autoencoder_logs')

    trainer.fit(
        model,
        torch.utils.data.DataLoader(train_dataset,
                                    batch_size=FLAGS.batch_size),
        torch.utils.data.DataLoader(validation_dataset,
                                    batch_size=FLAGS.batch_size))

    # Look at a few reconstructions after training
    model.to('cuda')
    model.eval()
    with torch.no_grad():
        for i in range(5):
            image, _ = validation_dataset[i]
            image = image.unsqueeze(0)
            image = image.view(1, 28 * 28)
            image = image.cuda()
            reconstructed_image = model(image).cpu()
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(image.cpu().squeeze(0).squeeze(0).view(28, 28),
                       cmap='gray')
            plt.title('Original image')
            plt.subplot(1, 2, 2)
            plt.imshow(reconstructed_image.squeeze(0).squeeze(0).view(28, 28),
                       cmap='gray')
            plt.title('Reconstructed image')
            plt.show()


if __name__ == '__main__':
    app.run(main)
