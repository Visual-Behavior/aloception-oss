import torch
from torch import nn

from alonet.multi_gpu import get_world_size, is_dist_avail_and_initialized
from alonet.detr import DetrCriterion
import torch.nn.functional as F
import aloscene


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


class PanopticCriterion(DetrCriterion):
    """This class computes the loss for DETR.
    The process happens in two steps:
    1) we compute hungarian assignment between ground truth boxes and the outputs of the model
    2) we supervise each pair of matched ground-truth / prediction (supervise class and box)"""

    def __init__(self, loss_dice_weight: float, loss_focal_weight: float, **kwargs):
        """Create the criterion.

        Parameters
        ----------
        num_classes: int
            number of object categories, omitting the special no-object category
        matcher: nn.Module
            module able to compute a matching between targets and proposed boxes
        loss_ce_weight: float
            Cross entropy class weight
        loss_boxes_weight: float
            Boxes loss l1 weight
        loss_giou_weight: float
            Boxes loss GIOU
        loss_dice_weight: float
            DICE/F-1 loss weight use in masks_loss
        loss_focal_weight: float
            Focal loss weight use in masks_loss
        eos_coef: float
            relative classification weight applied to the no-object category
        aux_loss_stage:
            Number of auxialiry stage
        losses: list
            list of all the losses to be applied. See get_loss for list of available losses.
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
    ) -> dict:
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
        if num_boxes == 0 or outputs["pred_masks"].numel() == 0:
            return {
                "loss_DICE": torch.tensor(0.0, device=frames.device, requires_grad=True),
                "loss_focal": torch.tensor(0.0, device=frames.device, requires_grad=True),
            }

        # Masks resize
        fr_shape = frames.shape[-2:]  # (H,W)
        outputs_masks = F.interpolate(outputs["pred_masks"], size=fr_shape, mode="bilinear", align_corners=False)

        # Masks alignment with indices
        pred_masks, target_masks = [], []
        zero_masks = torch.zeros_like(frames[0][0:1]).as_tensor()
        for gt_masks, p_masks, m_filters, b_index in zip(
            frames.segmentation, outputs_masks, outputs["pred_masks_info"]["filters"], indices
        ):
            # Get pred_masks by indices matcher and append zero mask if it is necessary
            m_index = torch.where(m_filters)[0]
            for ib in b_index[0].to(m_index.device):
                im = (ib == m_index).nonzero()
                if im.numel() > 0:
                    im = im.item()
                    pred_masks.append(p_masks[im : im + 1])
                else:
                    pred_masks.append(zero_masks)

            target_masks.append(gt_masks.as_tensor()[b_index[1]])

        # Reshape for loss process
        pred_masks = torch.cat(pred_masks, dim=0).flatten(1)
        target_masks = torch.cat(target_masks, dim=0).flatten(1).view(pred_masks.shape)

        # DICE/F-1 loss
        losses["loss_DICE"] = dice_loss(pred_masks, target_masks, num_boxes)

        # Sigmoid focal loss
        losses["loss_focal"] = sigmoid_focal_loss(pred_masks, target_masks, num_boxes, **kwargs)

        return losses

    def get_loss(
        self, loss: str, outputs: dict, frames: aloscene.Frame, indices: list, num_boxes: torch.Tensor, **kwargs
    ):
        return super().get_loss(loss, outputs, frames, indices, num_boxes, {"masks": self.loss_masks}, **kwargs)
