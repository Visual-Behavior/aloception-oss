"""`Pytorch Lightning Module <https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html>`_ to
train models based on :mod:`~alonet.detr_panoptic.detr_panoptic` module
"""

from alonet.detr_panoptic.utils import get_mask_queries, get_base_model_frame
import alonet


class LitPanopticDetr(alonet.detr.LitDetr):
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
        Name use to define the model, by default "detr-r50-panoptic"
    model : torch.nn, optional
        Custom model to train

    Notes
    -----
    Arguments entered by the user (kwargs) will replace those stored in args attribute
    """

    @staticmethod
    def add_argparse_args(parent_parser, parser=None):
        parser = parent_parser.add_argument_group("LitPanoptic") if parser is None else parser
        parser.add_argument(
            "--weights", type=str, default=None, help="One of (detr-r50-panoptic). Default: None",
        )
        parser.add_argument("--gradient_clip_val", type=float, default=0.1, help="Gradient clipping norm (default 0.1")
        parser.add_argument(
            "--accumulate_grad_batches", type=int, default=4, help="Number of gradient accumulation steps (default 4)"
        )
        parser.add_argument(
            "--model_name",
            type=str,
            default="detr-r50-panoptic",
            help="Model name to use. One of {'detr-r50-panoptic', 'deformable-detr-r50-panoptic'}"
            + " (default panoptic-detr)",
        )
        return parent_parser

    def training_step(self, frames, batch_idx):
        # Get correct set of labels and assert inputs content
        frames = get_base_model_frame(frames)
        self.assert_input(frames)
        get_filter_fn = lambda *args, **kwargs: get_mask_queries(
            *args, model=self.model.detr, matcher=self.matcher, **kwargs
        )
        m_outputs = self.model(frames, get_filter_fn=get_filter_fn)

        total_loss, losses = self.criterion(m_outputs, frames, compute_statistical_metrics=batch_idx < 100)

        outputs = {"loss": total_loss}
        outputs.update({"metrics": losses})
        outputs.update({"m_outputs": m_outputs})
        return outputs

    def validation_step(self, frames, batch_idx):
        # Get correct set of labels
        frames = get_base_model_frame(frames)
        return super().validation_step(frames, batch_idx)

    def build_model(self, num_classes=250, aux_loss=True, weights=None):
        """Build the default model

        Parameters
        ----------
        num_classes : int, optional
            Number of classes in embed layer, by default 250
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
            Only :attr:`detr-r50-panoptic` and :attr:`deformable-detr-r50-panoptic` models are supported yet.
        """
        if self.model_name == "detr-r50-panoptic":
            detr_model = alonet.detr.DetrR50Finetune(
                num_classes=num_classes, aux_loss=aux_loss, weights="detr-r50", background_class=250
            )
        elif self.model_name == "deformable-detr-r50-panoptic":
            detr_model = alonet.deformable_detr.DeformableDetrR50Refinement(
                num_classes=num_classes,
                aux_loss=aux_loss,
                weights="deformable-detr-r50-refinement",
                activation_fn="softmax",
                background_class=250,
            )
        else:
            raise Exception(f"Unsupported base model {self.model_name}")
        return alonet.detr_panoptic.PanopticHead(detr_model, weights=weights or self.weights)

    def build_criterion(
        self,
        matcher=None,
        loss_dice_weight=2,
        loss_focal_weight=2,
        loss_ce_weight=1,
        loss_boxes_weight=5,
        loss_giou_weight=2,
        eos_coef=0.1,
        losses=["masks", "boxes", "labels"],
        aux_loss_stage=6,
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
            Weight of DICE/F-1 loss in total loss, by default 2
        loss_focal_weight : float, optional
            Weight of sigmoid focal loss in total loss, by default 2
        eos_coef : float, optional
            Background/End of the Sequence (EOS) coefficient, by default 0.1
        losses : list, optional
            List of losses to take into account in total loss, by default ["labels", "boxes", "masks"].
            Possible values: ["labels", "boxes", "masks"] (use the latest in segmentation tasks)
        aux_loss_stage : int, optional
            Size of stages from :attr:`aux_outputs` key in forward ouputs, by default 6

        Returns
        -------
        :mod:`DetrCriterion <alonet.detr.criterion>`
            Criterion use to train the model
        """
        return alonet.detr.DetrCriterion(
            matcher=matcher or self.matcher,
            loss_ce_weight=loss_ce_weight,
            loss_boxes_weight=loss_boxes_weight,
            loss_giou_weight=loss_giou_weight,
            loss_dice_weight=loss_dice_weight,
            loss_focal_weight=loss_focal_weight,
            eos_coef=eos_coef,
            aux_loss_stage=aux_loss_stage,
            losses=losses,
        )

    def callbacks(self, data_loader):
        obj_detection_callback = alonet.detr_panoptic.PanopticObjectDetectorCallback(
            val_frames=next(iter(data_loader.val_dataloader()))
        )
        metrics_callback = alonet.callbacks.MetricsCallback()
        ap_metrics_callback = alonet.detr_panoptic.PanopticApMetricsCallbacks()
        pq_metrics_callback = alonet.callbacks.PQMetricsCallback()
        return [obj_detection_callback, metrics_callback, ap_metrics_callback, pq_metrics_callback]

    def run_train(self, data_loader, args, project="panoptic-detr", expe_name=None, callbacks: list = None):
        expe_name = expe_name or self.model_name
        super().run_train(data_loader, args, project, expe_name, callbacks)
