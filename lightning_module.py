"""Lightning module."""
import torch
import pytorch_lightning as pl
import torchmetrics
from torchmetrics import Accuracy


class MNISTClassifier(pl.LightningModule):

    def __init__(self, n_layers, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()

        model = torch.nn.Sequential(torch.nn.Flatten(),
                                    torch.nn.Linear(28 * 28, 128))
        for _ in range(n_layers):
            model.append(torch.nn.Linear(128, 128))
            model.append(torch.nn.ReLU())
        model.append(torch.nn.Linear(128, 10))

        self.model = model
        self.learning_rate = learning_rate

    def training_step(self, batch, _):
        loss, _ = self.compute_loss(batch)
        self.log('loss', loss, on_epoch=True, prog_bar=True)
        return {'loss': loss}

    def validation_step(self, batch, _):
        val_loss, val_accuracy = self.compute_loss(batch)
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True)
        self.log('val_accuracy', val_accuracy, on_epoch=True, prog_bar=True)
        return {'val_loss': val_loss, 'val_accuracy': val_accuracy}

    def compute_loss(self, batch):
        x, y = batch
        logits = self.model(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        accuracy_fn = Accuracy(task='multiclass', num_classes=10)
        accuracy = accuracy_fn(logits, y)
        return loss, accuracy

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class EncoderDecoderWrapper(pl.LightningModule):

    def __init__(self, latend_dim, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.encoder = nn.Sequential(
            torch.nn.Linear(784, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, latent_dim),
        )
        self.decoder = nn.Sequential(
            torch.nn.Linear(latent_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 784),
            torch.nn.Sigmoid(),
        )

    def training_step(self, batch, _):
        loss = self.compute_loss(batch)
        self.log('loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        loss = self.compute_loss(batch)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def compute_loss(self, batch):
        x, _ = batch
        z = self.encoder(x.view(x.size(0), -1))
        z = self.decoder(z)
        z = z.view(z.size(0), 1, 28, 28)
        loss = nn.MSELoss()(z, x)
        return loss

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class MNISTDifferentiableGenerator(pl.LightningModule):
    """Backpropagates with respect to the inputs to generate an image."""

    def __init__(self,
                 classifier,
                 autoencoder,
                 latent_dim,
                 class_to_generate,
                 learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=['classifier', 'autoencoder'])

        self.learning_rate = learning_rate
        self.class_to_generate = torch.tensor(class_to_generate)

        # self.latent = torch.nn.Parameter(torch.rand(1, latent_dim))
        self.latent = torch.nn.Parameter(
            torch.normal(torch.tensor([0.] * latent_dim),
                         torch.tensor([10.] * latent_dim)))

        self.classifier = classifier
        self.autoencoder = autoencoder
        self.classifier.freeze()
        self.autoencoder.freeze()

        self.latents = [self.latent.detach().cpu()]

    def training_step(self, _, __):
        image = self.autoencoder.decoder(self.latent).view(1, 28, 28)
        classifier_logits = self.classifier(image)
        classifier_probs = torch.nn.functional.log_softmax(classifier_logits,
                                                           dim=-1)
        mask_loss = torch.nn.functional.one_hot(
            self.class_to_generate,
            10).type_as(classifier_logits).to(classifier_probs.device)
        loss = torch.sum(-mask_loss * classifier_probs)
        self.log('loss', loss, on_step=True, prog_bar=True)
        self.latents.append(self.latent.detach().cpu())
        return loss

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optim
