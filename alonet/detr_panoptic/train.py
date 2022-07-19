"""`Pytorch Lightning Module <https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html>`_ to
train models based on :mod:`~alonet.detr_panoptic.detr_panoptic` module
"""

import alonet
import torch
from argparse import ArgumentParser, _ArgumentGroup, Namespace

from alonet.detr_panoptic.utils import get_mask_queries, get_base_model_frame
from aloscene import Frame


class LitPanopticDetr(alonet.deformable_detr.LitDeformableDetr):
    """
    Parameters
    ----------
    args : Namespace, optional
        Attributes stored in specific Namespace, by default None
    weights : str, optional
        Weights name to load, by default None
    gradient_clip_val : float, optional
        pytorch_lightning.trainer.trainer parameter. 0 means dont clip, by default 0.1
    accumulate_grad_batches : int, optional
        Accumulates grads every k batches or as set up in the dict, by default 4
    model_name : str, optional
        Name use to define the model, by default ``detr-r50-panoptic``
    model : torch.nn, optional
        Custom model to train

    Notes
    -----
    Arguments entered by the user (kwargs) will replace those stored in args attribute
    """

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser, parser: _ArgumentGroup = None):
        parser = parent_parser.add_argument_group("LitPanopticDetr") if parser is None else parser
        parser.add_argument(
            "--weights", type=str, default=None, help="One of {detr-r50-panoptic}, by default %(default)s",
        )
        parser.add_argument(
            "--gradient_clip_val", type=float, default=0.1, help="Gradient clipping norm, by default %(default)s"
        )
        parser.add_argument(
            "--accumulate_grad_batches",
            type=int,
            default=4,
            help="Number of gradient accumulation steps, by default %(default)s",
        )
        parser.add_argument("--track_grad_norm", type=int, default=-1, help="Tracks that p-norm, by default no track")
        parser.add_argument(
            "--model_name",
            type=str,
            default="detr-r50-panoptic",
            help="Model name to use. One of {'detr-r50-panoptic'}, by default %(default)s",
        )
        parser.add_argument(
            "--freeze_detr", action="store_true", help="Freeze DETR weights in training, by default %(default)s"
        )
        return parent_parser

    def training_step(self, frames: Frame, batch_idx: int):
        # Get correct set of labels and assert inputs content
        frames = get_base_model_frame(frames)
        self.assert_input(frames)
        get_filter_fn = lambda *args, **kwargs: get_mask_queries(
            *args, matcher=self.matcher, **kwargs
        )
        m_outputs = self.model(frames, get_filter_fn=get_filter_fn)

        total_loss, losses = self.criterion(m_outputs, frames, compute_statistical_metrics=batch_idx < 100)

        outputs = {"loss": total_loss}
        outputs.update({"metrics": losses})
        outputs.update({"m_outputs": m_outputs})
        return outputs

    def validation_step(self, frames: Frame, batch_idx: int):
        # Get correct set of labels
        frames = get_base_model_frame(frames)
        return super().validation_step(frames, batch_idx)

    def build_model(
        self, num_classes: int = 250, background_class: int = 250, aux_loss: bool = True, weights: str = None
    ):
        """Build the default model

        Parameters
        ----------
        num_classes : int, optional
            Number of classes in embed layer, by default 250
        background_class : int, optional
            Background class id, by default 250
        aux_loss : bool, optional
            Return auxiliar outputs in forward output, by default True
        weights : str, optional
            Path or id to load weights, by default None

        Returns
        -------
        :mod:`~alonet.detr_panoptic.detr_panoptic`
            Pytorch model

        Raises
        ------
        Exception
            Only ``detr-r50-panoptic`` models are supported yet.
        """
        if self.model_name == "detr-r50-panoptic":
            model = alonet.detr_panoptic.DetrR50Panoptic(
                num_classes=num_classes,
                aux_loss=aux_loss,
                background_class=background_class,
                weights=weights or self.weights,
                freeze_detr=self.freeze_detr,
            )
        else:
            raise Exception(f"Unsupported base model {self.model_name}")
        return model

    def build_criterion(
        self,
        matcher: torch.nn = None,
        loss_dice_weight: float = 2,
        loss_focal_weight: float = 2,
        loss_label_weight: float = 1,
        loss_boxes_weight: float = 5,
        loss_giou_weight: float = 2,
        eos_coef: float = 0.1,
        focal_alpha: float = 0.25,
        losses: list = ["masks", "boxes", "labels"],
        aux_loss_stage: int = 6,
    ):
        """Build the default criterion

        Parameters
        ----------
        matcher : torch.nn, optional
            One specfic matcher to use in criterion process, by default the output of :func:`build_matcher`
        loss_label_weight : float, optional
            Weight of cross entropy loss in total loss, by default 1
        loss_boxes_weight : float, optional
            Weight of boxes loss in total loss, by default 5
        loss_giou_weight : float, optional
            Weight of GIoU loss in total loss, by default 2
        loss_dice_weight : float, optional
            Weight of DICE/F-1 loss in total loss, by default 2
        loss_focal_weight : float, optional
            Weight of sigmoid focal loss in total loss, by default 2
        eos_coef : float, optional
            Background/End of the Sequence (EOS) coefficient, by default 0.1
        focal_alpha : float, optional
            This parameter is used only when the model use sigmoid activation function.
            Weighting factor in range (0,1) to balance positive vs negative examples. -1 for no weighting,
            by default 0.25
        losses : list, optional
            List of losses to take into account in total loss, by default [``labels``, ``boxes``, ``masks``] (uses all
            the possible values).
        aux_loss_stage : int, optional
            Size of stages from :attr:`aux_outputs` key in forward ouputs, by default 6

        Returns
        -------
        :mod:`DetrPanopticCriterion <alonet.detr_panoptic.criterion>`
            Criterion use to train the model
        """
        return alonet.detr_panoptic.DetrPanopticCriterion(
            matcher=matcher or self.matcher,
            loss_ce_weight=loss_label_weight,
            loss_boxes_weight=loss_boxes_weight,
            loss_giou_weight=loss_giou_weight,
            loss_dice_weight=loss_dice_weight,
            loss_focal_weight=loss_focal_weight,
            eos_coef=eos_coef,
            focal_alpha=focal_alpha,
            aux_loss_stage=aux_loss_stage,
            losses=losses,
        )

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

    def callbacks(self, data_loader: Frame):
        obj_detection_callback = alonet.detr_panoptic.PanopticObjectDetectorCallback(
            val_frames=next(iter(data_loader.val_dataloader()))
        )
        metrics_callback = alonet.callbacks.MetricsCallback()
        ap_metrics_callback = alonet.detr_panoptic.PanopticApMetricsCallbacks()
        pq_metrics_callback = alonet.callbacks.PQMetricsCallback()
        return [obj_detection_callback, metrics_callback, ap_metrics_callback, pq_metrics_callback]

    def run_train(
        self,
        data_loader: Frame,
        args: Namespace,
        project: str = "detr-panoptic",
        expe_name: str = None,
        callbacks: list = None,
    ):
        expe_name = expe_name or self.model_name
        super().run_train(data_loader, args, project, expe_name, callbacks)
