"""Callbacks"""
import pytorch_lightning as pl


class StopOnLoss(pl.Callback):

    def __init__(self, loss_threshold=.9):
        super().__init__()
        self.loss_threshold = loss_threshold

    def on_train_batch_end(self, trainer, pl_module, _, __, ___):
        if trainer.callback_metrics['loss'] < self.loss_threshold:
            print("Loss is below threshold, stopping training")
            trainer.should_stop = True
