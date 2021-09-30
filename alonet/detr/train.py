"""`Pytorch Lightning Module <https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html>`_ to
train models based on :mod:`~alonet.detr.detr` module
"""

import logging
import torch
from typing import Union

from argparse import ArgumentParser, Namespace
import pytorch_lightning as pl

from aloscene import Frame
import aloscene
import alonet


class LitDetr(pl.LightningModule):
    """
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

    def __init__(self, args: Namespace = None, model: torch.nn = None, **kwargs):
        super().__init__()
        # Update class attributes with args and kwargs inputs
        alonet.common.pl_helpers.params_update(self, args, kwargs)
        self._init_kwargs_config.update({"model": model})

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
        """Add arguments to parent parser with default values

        Parameters
        ----------
        parent_parser : ArgumentParser
            Object to append new arguments
        parser : ArgumentParser.argument_group, optional
            Argument group to append the parameters, by default None

        Returns
        -------
        ArgumentParser
            Object with new arguments concatenated
        """
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

    def forward(self, frames: Union[list, Frame]):
        """Run a forward pass through the model.

        Parameters
        ----------
        frames : Union[list, :mod:`Frames <aloscene.frame>`]
            List of :mod:`~aloscene.frame` without batch dimension or a Frame with the batch dimension

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
        """Given the model forward outputs, this method will return an
        :mod:`BoundingBoxes2D <aloscene.bounding_boxes_2d>` tensor.

        Parameters
        ----------
        m_outputs: dict
            Dict with the model forward outptus

        Returns
        -------
        List[:mod:`BoundingBoxes2D <aloscene.bounding_boxes_2d>`]
            Set of boxes for each batch
        """
        return self.model.inference(m_outputs)

    def training_step(self, frames: Union[list, Frame], batch_idx: int):
        """Train the model for one step

        Parameters
        ----------
        frames : Union[list, :mod:`Frames <aloscene.frame>`]
            List of :mod:`~aloscene.frame` without batch dimension or a Frame with the batch dimension
        batch_idx : int
            Batch id given by Lightning

        Returns
        -------
        dict
            Dictionary with the :attr:`loss` to optimize, :attr:`m_outputs` forward outputs and :attr:`metrics` to log.
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

    @torch.no_grad()
    def validation_step(self, frames: Union[list, Frame], batch_idx: int):
        """Run one step of validation

        Parameters
        ----------
        frames : Union[list, :mod:`Frames <aloscene.frame>`]
            List of :mod:`~aloscene.frame` without batch dimension or a Frame with the batch dimension
        batch_idx : int
            Batch id given by Lightning

        Returns
        -------
        dict
            Dictionary with the :attr:`loss` to optimize, :attr:`m_outputs` forward outputs and :attr:`metrics` to log.
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
        """AdamW optimizer configuration, using different learning rates for backbone and others parameters

        Returns
        -------
        torch.optim
            `AdamW <https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html>`_ optimizer to update weights
        """
        param_dicts = [
            {"params": [p for n, p in self.model.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.model.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": 1e-5,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=1e-4, weight_decay=1e-4)
        return optimizer

    def build_model(self, num_classes: int = 91, aux_loss: bool = True, weights: str = None):
        """Build the default model

        Parameters
        ----------
        num_classes : int, optional
            Number of classes in embed layer, by default 91
        aux_loss : bool, optional
            Return auxiliar outputs in forward output, by default True
        weights : str, optional
            Path or id to load weights, by default None

        Returns
        -------
        :mod:`~alonet.detr.detr`
            Pytorch model

        Raises
        ------
        Exception
            Only :attr:`detr-r50` models are supported yet.
        """
        if self.model_name == "detr-r50":
            return alonet.detr.DetrR50(num_classes=num_classes, aux_loss=aux_loss, weights=self.weights)
        else:
            raise Exception(f"Unsupported base model {self.model_name}")

    def build_matcher(self, cost_class: float = 1, cost_boxes: float = 5, cost_giou: float = 2):
        """Build the default matcher

        Parameters
        ----------
        cost_class : float, optional
            Weight of class cost in Hungarian Matcher, by default 1
        cost_boxes : float, optional
            Weight of boxes cost in Hungarian Matcher, by default 5
        cost_giou : float, optional
            Weight of GIoU cost in Hungarian Matcher, by default 2

        Returns
        -------
        :mod:`DetrHungarianMatcher <alonet.detr.matcher>`
            Hungarian Matcher, as a Pytorch model
        """
        return alonet.detr.DetrHungarianMatcher(cost_class=cost_class, cost_boxes=cost_boxes, cost_giou=cost_giou)

    def build_criterion(
        self,
        matcher: torch.nn = None,
        loss_ce_weight: float = 1,
        loss_boxes_weight: float = 5,
        loss_giou_weight: float = 2,
        loss_dice_weight: float = 0,
        loss_focal_weight: float = 0,
        eos_coef: float = 0.1,
        losses: list = ["labels", "boxes"],
        aux_loss_stage: int = 6,
    ):
        """Build the default criterion

        Parameters
        ----------
        matcher : torch.nn, optional
            One specfic matcher to use in criterion process, by default the output of :func:`build_matcher`
        loss_ce_weight : float, optional
            Weight of cross entropy loss in total loss, by default 1
        loss_boxes_weight : float, optional
            Weight of boxes loss in total loss, by default 5
        loss_giou_weight : float, optional
            Weight of GIoU loss in total loss, by default 2
        loss_dice_weight : float, optional
            Weight of DICE/F-1 loss in total loss, by default 0 (use by loss_masks procedure)
        loss_focal_weight : float, optional
            Weight of sigmoid focal loss in total loss, by default 0 (use by loss_masks procedure)
        eos_coef : float, optional
            Background/End of the Sequence (EOS) coefficient, by default 0.1
        losses : list, optional
            List of losses to take into account in total loss, by default ["labels", "boxes"].
            Possible values: ["labels", "boxes", "masks"] (use the latest in segmentation tasks)
        aux_loss_stage : int, optional
            Size of stages from :attr:`aux_outputs` key in forward ouputs, by default 6

        Returns
        -------
        :mod:`DetrCriterion <alonet.detr.criterion>`
            Criterion use to train the model
        """
        return alonet.detr.DetrCriterion(
            matcher=matcher,
            loss_ce_weight=loss_ce_weight,
            loss_boxes_weight=loss_boxes_weight,
            loss_giou_weight=loss_giou_weight,
            loss_dice_weight=loss_dice_weight,
            loss_focal_weight=loss_focal_weight,
            eos_coef=eos_coef,
            losses=losses,
            aux_loss_stage=aux_loss_stage,
        )

    def assert_input(self, frames: Frame, inference=False):
        """Check if input-frames have the correct format

        Parameters
        ----------
        frames : :mod:`Frames <aloscene.frame>`
            Input frames
        inference : bool, optional
            Check input from inference procedure, by default False
        """
        assert isinstance(frames, aloscene.Frame)
        assert frames.normalization == "resnet", f"{frames.normalization}"
        assert frames.mean_std[0] == self.model.INPUT_MEAN_STD[0]
        assert frames.mean_std[1] == self.model.INPUT_MEAN_STD[1]
        assert frames.names == ("B", "C", "H", "W"), f"{frames.names}"
        if not inference:
            assert frames.boxes2d is not None
            assert frames.mask is not None
            assert frames.mask.names == ("B", "C", "H", "W")

    def callbacks(self, data_loader: torch.utils.data.DataLoader):
        """Given a data loader, this method will return the default callbacks of the training loop.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            Dataloader to get a sample to use on :mod:`~alonet.callbacks.object_detector_callback`

        Returns
        -------
        List[:doc:`alonet.callbacks`]
            Callbacks use in train process
        """

        obj_detection_callback = alonet.detr.DetrObjectDetectorCallback(
            val_frames=next(iter(data_loader.val_dataloader()))
        )
        metrics_callback = alonet.callbacks.MetricsCallback()
        ap_metrics_callback = alonet.callbacks.ApMetricsCallback()
        return [obj_detection_callback, metrics_callback, ap_metrics_callback]

    def run_train(
        self,
        data_loader: torch.utils.data.DataLoader,
        args: Namespace = None,
        project: str = "detr",
        expe_name: str = "detr_50",
        callbacks: list = None,
    ):
        """Train the model using pytorch lightning

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            Dataloader use in :func:`callbacks` function
        project : str, optional
            Project name using to save checkpoints, by default "detr"
        expe_name : str, optional
            Specific experiment name to save checkpoints, by default "detr_50"
        callbacks : list, optional
            List of callbacks to use, by default :func:`callbacks` output
        args : Namespace, optional
            Additional arguments use in training process, by default None
        """
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
        level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s", datefmt="%d-%m-%y %H:%M:%S",
    )
    logger = logging.getLogger("aloception")

    # Random inference
    model = LitDetr(args)
    frame = Frame(torch.rand((3, 250, 250)), names=("C", "H", "W")).norm_resnet()
    frame = frame.batch_list([frame])
    logger.info("Random inference without train: {}".format(model.inference(model(frame))))
