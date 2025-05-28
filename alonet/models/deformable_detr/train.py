import torch

from argparse import ArgumentParser
from typing import Union

from alonet.deformable_detr import DeformableDetrR50, DeformableDetrR50Refinement
from alonet.deformable_detr import DeformableCriterion, DeformableDetrHungarianMatcher
from alonet.detr import LitDetr


class LitDeformableDetr(LitDetr):
    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser, parser=None):
        """Add argument for Deformable DETR

        Parameters
        ----------
        parent_parser : ArgumentParser
            parser to be added new argument
        parser : [type], optional
            parser to overwrite parent_parser if not None, by default None

        Returns
        -------
        ArgumentParser
            Parser with arguments for Deformable DETR added
        """
        parser = parent_parser.add_argument_group("LitDeformableDetr") if parser is None else parser
        parser.add_argument(
            "--weights",
            type=str,
            default=None,
            help="One of (deformable-detr-r50-refinement, deformable-detr-r50). Default: None",
        )
        parser.add_argument("--gradient_clip_val", type=float, default=0.1, help="Gradient clipping norm (default 0.1")
        parser.add_argument(
            "--accumulate_grad_batches", type=int, default=4, help="Number of gradient accumulation steps (default 4)"
        )
        parser.add_argument(
            "--track_grad_norm", type=int, default=-1, help="Default -1, no track. Otherwise tracks that p-norm."
        )
        parser.add_argument(
            "--model_name",
            type=str,
            default="deformable-detr-r50-refinement",
            help="Model name to use. One of ['deformable-detr-r50-refinement', 'deformable-detr-r50']. "
            + "(default deformable-detr-r50-refinement)",
        )

        return parent_parser

    def build_model(
        self, num_classes=91, aux_loss=True, weights=None, activation_fn="sigmoid"
    ) -> Union[DeformableDetrR50Refinement, DeformableDetrR50]:
        """Build model for training

        Parameters
        ----------
        num_classes : int, optional
            Number of classes to detect, by default 91
        aux_loss : bool, optional
            aux_loss : bool, optional
            If True, the model will returns auxilary outputs at each decoder layer
            to calculate auxiliary decoding losses. By default True.
        weights : str, optional
            Pretrained weights, by default None
        activation_fn : str, optional
            Activation function for classification head. Either "sigmoid" or "softmax". By default "sigmoid".

        Returns
        -------
        Union[DeformableDetrR50Refinement, DeformableDetrR50]
            Deformable DETR for training
        """
        if self.model_name == "deformable-detr-r50-refinement":
            return DeformableDetrR50Refinement(
                num_classes=num_classes, aux_loss=aux_loss, weights=self.weights, activation_fn=activation_fn
            )
        elif self.model_name == "deformable-detr-r50":
            return DeformableDetrR50(
                num_classes=num_classes, aux_loss=aux_loss, weights=self.weights, activation_fn=activation_fn
            )
        else:
            raise Exception(f"Unsupported base model {self.model_name}")

    def build_criterion(
        self,
        matcher: DeformableDetrHungarianMatcher,
        loss_label_weight=1,
        loss_boxes_weight=5,
        loss_giou_weight=2,
        losses=["labels", "boxes"],
        aux_loss_stage=6,
        eos_coef=0.1,
    ) -> DeformableCriterion:
        """Build criterion module to calculate losses

        Parameters
        ----------
        matcher : DeformableDetrHungarianMatcher
            Hungarian matcher to match between predictions and targets
        loss_label_weight : int, optional
            Weight of the classification loss, by default 1
        loss_boxes_weight : int, optional
            Weight of the L1 loss of the bounding box coordinates, by default 5
        loss_giou_weight : int, optional
            Weight of the giou loss of the bounding box, by default 2
        losses : list, optional
            Type of loss use in training, by default ['labels', 'boxes']
        aux_loss_stage : int, optional
            Number of auxiliary decoder stages, by default 6
        eos_coef : float, optional
            Relative classification weight applied to the no-object category, by default 0.1.
            This factor is applied only when softmax activation is used in model.

        Returns
        -------
        DeformableCriterion
        """
        return DeformableCriterion(
            matcher=matcher,
            loss_label_weight=loss_label_weight,
            loss_boxes_weight=loss_boxes_weight,
            loss_giou_weight=loss_giou_weight,
            losses=losses,
            aux_loss_stage=aux_loss_stage,
            eos_coef=eos_coef,
        )

    def build_matcher(self, cost_class=1, cost_boxes=5, cost_giou=2) -> DeformableDetrHungarianMatcher:
        """Build matcher to match between predictions and targets

        Parameters
        ----------
        cost_class : int, optional
            Weight of the classification error in the matching cost, by default 1
        cost_boxes : int, optional
            Weight of the L1 error of the bounding box coordinates in the matching cost, by default 5
        cost_giou : int, optional
            Weight of the giou loss of the bounding box in the matching cost, by default 2

        Returns
        -------
        DeformableDetrHungarianMatcher
        """
        return DeformableDetrHungarianMatcher(cost_class=cost_class, cost_boxes=cost_boxes, cost_giou=cost_giou)

    def configure_optimizers(self):
        """Configure optimzier using AdamW"""
        match_name_keywords = lambda name, keywords: any([key in name for key in keywords])

        linear_proj_names = ["reference_points", "sampling_offsets"]
        param_dicts = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if match_name_keywords(n, linear_proj_names) and p.requires_grad
                ],
                "lr": 1e-5,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": 1e-5,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if ("backbone" not in n) and (not match_name_keywords(n, linear_proj_names)) and p.requires_grad
                ]
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=1e-4, weight_decay=1e-4)
        return optimizer


if __name__ == "__main__":
    args = LitDeformableDetr.add_argparse_args(ArgumentParser()).parse_args()  # Help provider
    model = LitDeformableDetr(args)
