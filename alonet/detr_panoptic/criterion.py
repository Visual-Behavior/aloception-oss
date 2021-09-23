# Inspired by the official DETR repository and adapted for aloception
# https://github.com/facebookresearch/detr/blob/eb9f7e03ed8e2ed2cd55528989fe7df890bc3fc0/models/segmentation.py

from typing import Dict

import torch
import torch.nn.functional as F

import aloscene
import alonet


def dice_loss(inputs: torch.Tensor, targets: torch.Tensor, num_boxes: int) -> torch.Tensor:
    """Compute the DICE loss, similar to generalized IOU for masks

    Parameters
    ----------
    inputs : [type]
        A float tensor of arbitrary shape. The predictions for each example.
    targets : [type]
        A float tensor with the same shape as inputs. Stores the binary classification label for each element in inputs
        (0 for the negative class and 1 for the positive class).
    num_boxes : [type]
        Number of boxes
    Returns
    -------
    torch.Tensor
        DICE/F-1 loss
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


def sigmoid_focal_loss(
    inputs: torch.Tensor, targets: torch.Tensor, num_boxes: int, alpha: float = 0.25, gamma: float = 2
) -> torch.Tensor:
    """Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Parameters
    ----------
    inputs : [type]
        A float tensor of arbitrary shape. The predictions for each example.
    targets : [type]
        A float tensor with the same shape as inputs. Stores the binary classification label for each element in inputs
        (0 for the negative class and 1 for the positive class).
    num_boxes : [type]
        Number of boxes
    alpha : float, optional
        (optional) Weighting factor in range (0,1) to balance positive vs negative examples, by default 0.25
    gamma : float, optional
        Exponent of the modulating factor (1 - p_t) to balance easy vs hard examples, by default 2

    Returns
    -------
    torch.Tensor
        Sigmoid focal loss
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


class PanopticCriterion(alonet.detr.DetrCriterion):
    """This class computes the loss for Panoptic.
    Each pair of matched ground-truth / prediction (supervised class and masks/boxes)"""

    def __init__(
        self, loss_dice_weight: float, loss_focal_weight: float, **kwargs,
    ):
        """Create the criterion.

        Parameters
        ----------
        matcher: nn.Module
            module able to compute a matching between targets and proposed boxes
        loss_DICE_f1_weight: float
            DICE/F-1 loss weight
        loss_focal_weight: float
            Focal loss weight
        aux_loss_stage:
            Number of auxialiry stage
        losses: list
            list of all the losses to be applied. See get_loss for list of available losses.
        kwargs: Dict
            Parameters for boxes-criterion (see DetrCriterion)
        """
        super().__init__(**kwargs)

        # Define the weight dict
        self.loss_weights.update({"loss_DICE": loss_dice_weight, "loss_focal": loss_focal_weight})

        if kwargs["aux_loss_stage"] > 0:
            for i in range(kwargs["aux_loss_stage"] - 1):
                self.loss_weights.update({f"loss_focal_{i}": loss_focal_weight})
                self.loss_weights.update({f"loss_DICE_{i}": loss_dice_weight})

    def loss_masks(
        self, outputs: dict, frames: aloscene.Frame, indices: list, num_boxes: torch.Tensor, **kwargs
    ) -> Dict:
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss

        Parameters
        ----------
        outputs: dict
            Detr model forward outputs
        frames: aloscene.Frane
            Trgat frame with boxes2d and labels
        indices: list
            List of tuple with predicted indices and target indices
        num_boxes: torch.Tensor
            Number of total target boxes

        Returns
        -------
        Dict:
            DICE and focal losses results
        """
        assert frames.names[0] == "B"
        assert frames.segmentation[0].labels is not None and frames.segmentation[0].labels.encoding == "id"

        losses = {}
        if num_boxes == 0:
            return {}

        target_masks = torch.cat(
            [
                # Select masks per batch following the `target_indices` from the Hungarian matching
                masks.as_tensor()[indices[b][1]]
                for b, masks in enumerate(frames.segmentation)
            ],
            dim=0,
        )

        # Masks resize
        outputs_masks = F.interpolate(
            outputs["pred_masks"], size=target_masks.shape[-2:], mode="bilinear", align_corners=False
        )

        # Masks alignment with indices
        pred_masks = []
        zero_masks = torch.zeros_like(target_masks[0:1])
        for b, (masks, m_filters) in enumerate(zip(outputs_masks, outputs["pred_masks_info"]["filters"])):
            m_index = torch.where(m_filters)[0].cpu()
            b_index = indices[b][0].cpu()
            masks = {ib.item(): masks[ib == m_index] for ib in b_index if ib in m_index}

            for ib in b_index:
                ib = ib.item()
                if ib not in masks:
                    masks[ib] = zero_masks.clone()
            if len(masks) > 0:
                masks = torch.cat([m[1] for m in sorted(masks.items(), key=lambda x: x[0])], dim=0)
            else:
                masks = zero_masks[[]].view(0, *target_masks.shape[-2:])

            pred_masks.append(masks)

        pred_masks = torch.cat(pred_masks, dim=0)

        # Reshape for loss process
        pred_masks = pred_masks.flatten(1)
        target_masks = target_masks.flatten(1).view(pred_masks.shape)

        # DICE/F-1 loss
        losses["loss_DICE"] = dice_loss(pred_masks, target_masks, num_boxes)

        # Sigmoid focal loss
        losses["loss_focal"] = sigmoid_focal_loss(pred_masks, target_masks, num_boxes, **kwargs)

        return losses

    def get_loss(
        self, loss: str, outputs: dict, frames: aloscene.Frame, indices: list, num_boxes: torch.Tensor, **kwargs
    ) -> Dict:
        """Compute a loss given the model outputs, the target frame, the results from the matcher
        and the number of total boxes accross the devices.

        Parameters
        ----------
        loss: str
            Name of the loss to compute
        outputs: dict
            Detr model forward outputs
        frames: aloscene.Frane
            Trgat frame with boxes2d and labels
        indices: list
            List of tuple with predicted indices and target indices
        num_boxes: torch.Tensor
            Number of total target boxes

        Returns
        -------
        Dict
            Losses of the loss procedure.
        """
        loss_map = {"labels": self.loss_labels, "boxes": self.loss_boxes, "masks": self.loss_masks}
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, frames, indices, num_boxes, **kwargs)
