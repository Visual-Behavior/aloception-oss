from alonet.detr_panoptic.utils import get_mask_queries
import alonet
import aloscene


class LitPanopticDetr(alonet.detr.LitDetr):
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
        Name use to define the model, by default "detr-r50-panoptic"
    model : torch.nn, optional
        Custom model to train

    Notes
    -----
    Arguments entered by the user (kwargs) will replace those stored in args attribute
    """

    @staticmethod
    def add_argparse_args(parent_parser, parser=None):
        """Add arguments to parent parser with default values"""
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
        get_filter_fn = lambda *args, **kwargs: get_mask_queries(
            *args, model=self.model.detr, matcher=self.matcher, **kwargs
        )
        m_outputs = self.model(frames, get_filter_fn=get_filter_fn)

        total_loss, losses = self.criterion(m_outputs, frames, compute_statistical_metrics=batch_idx < 100)

        outputs = {"loss": total_loss}
        outputs.update({"metrics": losses})
        outputs.update({"m_outputs": m_outputs})
        return outputs

    def build_model(self, num_classes=250, aux_loss=True, weights=None):
        """Build model with default parameters"""
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
        """Build default criterion"""
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
        """Default callbacks"""
        obj_detection_callback = alonet.detr_panoptic.PanopticObjectDetectorCallback(
            val_frames=next(iter(data_loader.val_dataloader()))
        )
        metrics_callback = alonet.callbacks.MetricsCallback()
        ap_metrics_callback = alonet.callbacks.ApMetricsCallback()
        return [obj_detection_callback, metrics_callback, ap_metrics_callback]

    def run_train(self, data_loader, args, project="panoptic-detr", expe_name=None, callbacks: list = None):
        expe_name = expe_name or self.model_name
        super().run_train(data_loader, args, project, expe_name, callbacks)
