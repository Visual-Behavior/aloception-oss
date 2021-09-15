from datetime import time
import os, logging
import torch
from torch import nn
import torch.nn.functional as F

from argparse import ArgumentParser, Namespace

import pytorch_lightning as pl
import wandb

from typing import *

from alonet.callbacks import MetricsCallback
from alonet.detr import CocoDetection2Detr
from aloscene import Frame

import alodataset
import aloscene
import alonet
from aloscene import frame


class LitDetr(pl.LightningModule):
    def __init__(self, args: Namespace = None, model: torch.nn = None, **kwargs):
        """LightningModule to train Detr models

        Parameters
        ----------
        args : Namespace, optional
            Attributes stored in specific Namespace, by default None
        weights : str, optional
            Weights name to load, by default None
        gradient_clip_val : float, optional
            pytorch_lightning.trainer.trainer parameter. 0 means don’t clip, by default 0.1
        accumulate_grad_batches : int, optional
            Accumulates grads every k batches or as set up in the dict, by default 4
        model_name : str, optional
            Name use to define the model, by default "detr-r50"
        model : torch.nn, optional
            Custom model to train

        Notes
        -----
        Arguments entered by the user (kwargs) will replace those stored in args attribute
        """
        super().__init__()
        # Update class attributes with args and kwargs inputs
        alonet.common.pl_helpers.params_update(self, args, kwargs)

        # Load model
        if model is not None:
            if isinstance(model, str):
                self.model_name = model
                self.model = self.build_model()
            elif self.weights is None:
                self.model = model
            else:
                raise Exception(f"Weights of custom model doesnt match with {self.weights} weights")
        else:
            self.model = self.build_model()
        # Buld matcher
        self.matcher = self.build_matcher()
        # Build criterion
        self.criterion = self.build_criterion(matcher=self.matcher)

    @staticmethod
    def add_argparse_args(parent_parser, parser=None):
        parser = parent_parser.add_argument_group("LitDetr") if parser is None else parser
        parser.add_argument("--weights", type=str, default=None, help="One of (detr-r50). Default: None")
        parser.add_argument("--gradient_clip_val", type=float, default=0.1, help="Gradient clipping norm (default 0.1")
        parser.add_argument(
            "--accumulate_grad_batches", type=int, default=4, help="Number of gradient accumulation steps (default 4)"
        )
        parser.add_argument(
            "--model_name",
            type=str,
            default="detr-r50",
            help="Model name to use. One of ['detr-r50']. (default detr-r50)",
        )
        return parent_parser

    def forward(self, frames: Union[list, aloscene.Frame]):
        """Run a forward pass through the model.

        Parameters
        ----------
        frames: list | aloscene.Frame
            List of aloscene.Frame without batch dimension or a Frame with the batch dimension

        Returns
        -------
        m_outputs: dict
            A dict with the forward detr outputs.
        """
        # Batch list of frame if needed
        if isinstance(frames, list):
            frames = aloscene.Frame.batch_list(frames)
        # Assert inputs content
        self.assert_input(frames, inference=True)
        # Run forward pass
        m_outputs = self.model(frames)
        return m_outputs

    def inference(self, m_outputs: dict):
        """Given the model forward outputs, this method
        will retrun an aloscene.BoundingBoxes2D tensor.

        Parameters
        ----------
        m_outputs: dict
            Dict with the model forward outptus

        Returns
        -------
        boxes: aloscene.BoundingBoxes2D
        """
        return self.model.inference(m_outputs)

    def training_step(self, frames, batch_idx):
        """Train the model for one step

        Parameters
        ----------
        frames: list | aloscene.Frame
            List of aloscene.Frame without batch dimension or a Frame with the batch dimension
        batch_idx: int
            Batch id given by Lightning

        Returns
        -------
        outptus: dict
            dict with the `loss` to optimize and the `metrics` to log.
        """
        # Batch list of frame if needed
        if isinstance(frames, list):
            frames = aloscene.Frame.batch_list(frames)

        # Assert inputs content
        self.assert_input(frames)
        m_outputs = self.model(frames)

        total_loss, losses = self.criterion(m_outputs, frames, compute_statistical_metrics=batch_idx < 100)

        outputs = {"loss": total_loss}
        outputs.update({"metrics": losses})
        outputs.update({"m_outputs": m_outputs})
        return outputs

    def validation_step(self, frames, batch_idx):
        """Run one step of validation

        Parameters
        ----------
        frames: list | aloscene.Frame
            List of aloscene.Frame without batch dimension or a Frame with the batch dimension
        batch_idx: int
            Batch id given by Lightning

        Returns
        -------
        outptus: dict
            dict with the `val_loss` and the metrics to log
        """
        # Batch list of frame if needed
        if isinstance(frames, list):
            frames = aloscene.Frame.batch_list(frames)

        # Assert inputs content
        self.assert_input(frames)
        m_outputs = self.model(frames)

        total_loss, losses = self.criterion(m_outputs, frames, compute_statistical_metrics=batch_idx < 100)

        self.log("val_loss", total_loss)
        outputs = {"val_loss": total_loss}
        outputs.update({"metrics": losses})
        outputs.update({"m_outputs": m_outputs})

        return outputs

    def configure_optimizers(self):
        """Configure optimzier using AdamW"""
        param_dicts = [
            {"params": [p for n, p in self.model.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.model.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": 1e-5,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=1e-4, weight_decay=1e-4)
        return optimizer

    def build_model(self, num_classes=91, aux_loss=True, weights=None):
        if self.model_name == "detr-r50":
            return alonet.detr.DetrR50(num_classes=num_classes, aux_loss=aux_loss, weights=self.weights)
        else:
            raise Exception(f"Unsupported base model {self.model_name}")

    def build_matcher(self, cost_class=1, cost_boxes=5, cost_giou=2):
        return alonet.detr.DetrHungarianMatcher(cost_class=cost_class, cost_boxes=cost_boxes, cost_giou=cost_giou)

    def build_criterion(
        self,
        matcher=None,
        loss_ce_weight=1,
        loss_boxes_weight=5,
        loss_giou_weight=2,
        eos_coef=0.1,
        losses=["labels", "boxes"],
        aux_loss_stage=6,
    ):
        return alonet.detr.DetrCriterion(
            matcher=matcher,
            loss_ce_weight=loss_ce_weight,
            loss_boxes_weight=loss_boxes_weight,
            loss_giou_weight=loss_giou_weight,
            eos_coef=eos_coef,
            losses=losses,
            aux_loss_stage=aux_loss_stage,
        )

    def assert_input(self, frames, inference=False):
        assert isinstance(frames, aloscene.Frame)
        assert frames.normalization == "resnet", f"{frames.normalization}"
        assert frames.mean_std[0] == self.model.INPUT_MEAN_STD[0]
        assert frames.mean_std[1] == self.model.INPUT_MEAN_STD[1]
        assert frames.names == ("B", "C", "H", "W"), f"{frames.names}"
        if not inference:
            assert frames.boxes2d is not None
            assert frames.mask is not None
            assert frames.mask.names == ("B", "C", "H", "W")

    def callbacks(self, data_loader):
        """Given a data loader, this method will return the default callbacks
        of the training loop.
        """

        obj_detection_callback = alonet.detr.DetrObjectDetectorCallback(
            val_frames=next(iter(data_loader.val_dataloader()))
        )
        metrics_callback = alonet.callbacks.MetricsCallback()
        ap_metrics_callback = alonet.callbacks.ApMetricsCallback()

        return [obj_detection_callback, metrics_callback, ap_metrics_callback]

    def run_train(self, data_loader, args=None, project="detr", expe_name="detr_50", callbacks: list = None):
        """Train the model using pytorch lightning"""
        # Set the default callbacks if not provide.
        callbacks = callbacks if callbacks is not None else self.callbacks(data_loader)

        alonet.common.pl_helpers.run_pl_training(
            # Trainer, data & callbacks
            lit_model=self,
            data_loader=data_loader,
            callbacks=callbacks,
            # Project info
            args=args,
            project=project,
            expe_name=expe_name,
        )


if __name__ == "__main__":
    args = LitDetr.add_argparse_args(ArgumentParser()).parse_args()  # Help provider

    # Logger config
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%d-%m-%y %H:%M:%S",
    )
    logger = logging.getLogger("aloception")

    # Random inference
    model = LitDetr(args)
    frame = Frame(torch.rand((3, 250, 250)), names=("C", "H", "W")).norm_resnet()
    frame = frame.batch_list([frame])
    logger.info("Random inference without train: {}".format(model.inference(model(frame))))
