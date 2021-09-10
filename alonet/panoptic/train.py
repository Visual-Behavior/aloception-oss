import alonet


class LitPanopticDetr(alonet.detr.LitDetr):
    @staticmethod
    def add_argparse_args(parent_parser, parser=None):
        parser = parent_parser.add_argument_group("LitPanoptic") if parser is None else parser
        parser.add_argument(
            "--weights",
            type=str,
            default=None,
            help="One of (panoptic-detr-r50, panoptic-deformable-detr-r50). Default: None",
        )
        parser.add_argument("--gradient_clip_val", type=float, default=0.1, help="Gradient clipping norm (default 0.1")
        parser.add_argument(
            "--accumulate_grad_batches", type=int, default=4, help="Number of gradient accumulation steps (default 4)"
        )
        parser.add_argument(
            "--model_name",
            type=str,
            default="panoptic-detr-r50",
            help="Model name to use. One of ['panoptic-detr-r50', 'panoptic-deformable-detr-r50']"
            + " (default panoptic-detr)",
        )
        return parent_parser

    def build_model(self, num_classes=184, aux_loss=True, weights=None):
        if self.model_name == "panoptic-detr-r50":
            detr_model = alonet.detr.DetrR50Finetune(num_classes=num_classes, aux_loss=aux_loss, weights="detr-r50")
        elif self.model_name == "panoptic-deformable-detr-r50":
            detr_model = alonet.deformable_detr.DeformableDetrR50RefinementFinetune(
                num_classes=num_classes,
                aux_loss=aux_loss,
                weights="deformable-detr-r50-refinement",
                activation_fn="softmax",
            )
        else:
            raise Exception(f"Unsupported base model {self.model_name}")
        return alonet.panoptic.PanopticHead(detr_model)

    def build_matcher(self):
        return alonet.panoptic.PanopticHungarianMatcher()

    def build_criterion(
        self, matcher=None, loss_dice_weight=1, loss_focal_weight=1, losses=["masks"], aux_loss_stage=6,
    ):
        return alonet.panoptic.PanopticCriterion(
            matcher=matcher or self.matcher,
            loss_dice_weight=loss_dice_weight,
            loss_focal_weight=loss_focal_weight,
            aux_loss_stage=aux_loss_stage,
            losses=losses,
        )

    def callbacks(self, data_loader):
        metrics_callback = alonet.callbacks.MetricsCallback()
        # obj_detection_callback = alonet.detr.DetrObjectDetectorCallback(
        #     val_frames=next(iter(data_loader.val_dataloader()))
        # )
        return [metrics_callback]

    def run_train(self, data_loader, args, project="panoptic-detr", expe_name=None, callbacks: list = None):
        expe_name = expe_name or self.model_name
        super().run_train(data_loader, args, project, expe_name, callbacks)
