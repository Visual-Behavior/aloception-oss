"""This class computes the loss for
:mod:`DEFORMABLE DETR PANOPTIC <alonet.deformable_detr_panoptic.deformable_detr_panoptic>`.
The process happens in two steps:

1) We compute hungarian assignment between ground truth boxes and the outputs of the model
2) We supervise each pair of matched ground-truth / prediction (supervise class, boxes and masks).
"""

from alonet.detr_panoptic.criterion import PanopticCriterion
from alonet.deformable_detr import DeformableCriterion


class DeformablePanopticCriterion(PanopticCriterion, DeformableCriterion):
    """Create the criterion.

    Parameters
    ----------
    num_classes: int
        number of object categories, omitting the special no-object category
    matcher: nn.Module
        module able to compute a matching between targets and proposed boxes
    loss_label_weight : float
        Label class weight, to use in CE or sigmoid focal (default) loss (depends of network configuration)
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
    focal_alpha : float, optional
        This parameter is used only when the model use sigmoid activation function.
        Weighting factor in range (0,1) to balance positive vs negative examples. -1 for no weighting, by default 0.25
    aux_loss_stage:
        Number of auxialiry stage
    losses: list
        list of all the losses to be applied. See get_loss for list of available losses.
    """
