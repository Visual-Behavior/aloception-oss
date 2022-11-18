"""Callback for any object detection training that use :mod:`Frame <aloscene.frame>` as GT. """
import aloscene
import numpy as np
import pytorch_lightning as pl
import torch
import wandb

from alodataset.utils.panoptic_utils import VOID_CLASS_ID
from alonet.common.logger import log_image, log_hyperparams
from argparse import ArgumentParser
from pytorch_lightning.utilities import rank_zero_only
from typing import Union


class HyperParametersCallback(pl.Callback):
    """The callback load frames every x training step as well as once every validation step on the given
    :attr:`val_frames` and log the different objects predicted

    Parameters
    ----------
    val_frames : Union[list, :mod:`Frames <aloscene.frame>`]
        List of sample from the validation set to use to load the validation progress
    """

    def __init__(self, val_frames: Union[list, aloscene.Frame], one_color_per_class: bool = True):
        """The callback load frames every x training step as well as once
        every validation step on the given `val_frames`

        Parameters
        ----------
        val_frames: list of :mod:`~aloscene.frame`
            List of sample from the validation set to use to load the validation progress
        one_color_per_class
            Set same segmentation-color for all objects with same category ID, by default True
        """
        super().__init__()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        print("HYPERPARM CALLBACK")
        if trainer.logger is None:
            return
        args = pl_module.add_argparse_args(
            ArgumentParser()).parse_args()
        wandb.config.update(args)  # adds all of the arguments as config variables
        log_hyperparams(trainer, wandb.config)

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        print("HYPERPARM CALLBACK")

        if trainer.logger is None:
            return
        args = pl_module.add_argparse_args(
            ArgumentParser()).parse_args()
        wandb.config.update(args)  # adds all of the arguments as config variables
        log_hyperparams(trainer, wandb.config)
