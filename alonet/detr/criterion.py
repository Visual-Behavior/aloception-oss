"""This class computes the loss for :mod:`DETR <alonet.detr.detr>`. The process happens in two steps:

1) We compute hungarian assignment between ground truth boxes and the outputs of the model
2) We supervise each pair of matched ground-truth / prediction (supervise class and box).
"""
import torch
from torch import nn

from alonet.multi_gpu import get_world_size, is_dist_avail_and_initialized
import torch.nn.functional as F
import aloscene


def dice_loss(inputs: torch.Tensor, targets: torch.Tensor, num_boxes: int):
    """Compute the DICE loss, similar to generalized IOU for masks

    Parameters
    ----------
    inputs : torch.Tensor
        A float tensor of arbitrary shape. The predictions for each example.
    targets : torch.Tensor
        A float tensor with the same shape as inputs. Stores the binary classification label for each element in inputs
        (0 for the negative class and 1 for the positive class).
    num_boxes : int
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
):
    """Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Parameters
    ----------
    inputs : torch.Tensor
        A float tensor of arbitrary shape. The predictions for each example.
    targets : torch.Tensor
        A float tensor with the same shape as inputs. Stores the binary classification label for each element in inputs
        (0 for the negative class and 1 for the positive class).
    num_boxes : int
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


class DetrCriterion(nn.Module):
    """
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

    def __init__(
        self,
        matcher: nn.Module,
        loss_ce_weight: float,
        loss_boxes_weight: float,
        loss_giou_weight: float,
        loss_dice_weight: float,
        loss_focal_weight: float,
        eos_coef: float,
        aux_loss_stage: int,
        losses: list,
    ):
        super().__init__()
        self.matcher = matcher
        self.eos_coef = eos_coef
        self.losses = losses

        # Define the weight dict
        loss_weights = {
            "loss_ce": loss_ce_weight,
            "loss_bbox": loss_boxes_weight,
            "loss_giou": loss_giou_weight,
            "loss_DICE": loss_dice_weight,
            "loss_focal": loss_focal_weight,
        }
        if aux_loss_stage > 0:
            aux_loss_weights = {}
            for i in range(aux_loss_stage - 1):
                aux_loss_weights.update({k + f"_{i}": v for k, v in loss_weights.items()})
            loss_weights.update(aux_loss_weights)
        self.loss_weights = loss_weights

    def loss_labels(self, outputs: dict, frames: aloscene.Frame, indices: list, num_boxes: torch.Tensor, **kwargs):
        """Compute the clasification loss

        Parameters
        ----------
        outputs : dict
            Detr model forward outputs
        frames : :mod:`Frames <aloscene.frame>`
            Target frame with boxes2d and labels
        indices : list
            List of tuple with predicted indices and target indices
        num_boxes : torch.Tensor
            Number of total target boxes
        """
        assert frames.names[0] == "B"
        assert frames.boxes2d[0].labels is not None and frames.boxes2d[0].labels.encoding == "id"

        background_class = len(frames.boxes2d[0].labels.labels_names)
        num_classes = len(frames.boxes2d[0].labels.labels_names) + 1

        # Select the target labels in each batch and concat everything
        target_classes_pos = torch.cat(
            [boxes2d.labels.as_tensor()[indices[b][1]] for b, boxes2d in enumerate(frames.boxes2d)]
        )
        target_classes_pos = target_classes_pos.type(torch.long)

        pred_logits = outputs["pred_logits"]
        idx = self._get_src_permutation_idx(indices)

        target_classes = torch.full(
            pred_logits.shape[:2], background_class, dtype=torch.int64, device=pred_logits.device
        )
        target_classes[idx] = target_classes_pos

        empty_weight = torch.ones(num_classes, device=target_classes.device)
        empty_weight[background_class] = self.eos_coef

        loss_ce = F.cross_entropy(pred_logits.transpose(1, 2), target_classes, empty_weight)
        losses = {"loss_ce": loss_ce}

        return losses

    def loss_boxes(self, outputs: dict, frames: aloscene.Frame, indices: list, num_boxes: torch.Tensor, **kwargs):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss

        Parameters
        ----------
        outputs : dict
            Detr model forward outputs
        frames : :mod:`Frames <aloscene.frame>`
            Target frame with boxes2d and labels
        indices : list
            List of tuple with predicted indices and target indices
        num_boxes : torch.Tensor
            Number of total target boxes
        """
        losses = {}
        if num_boxes == 0:
            return {}

        # print('loss_boxes:indices', indices)
        idx = self._get_src_permutation_idx(indices)

        pred_boxes = outputs["pred_boxes"][idx]
        pred_boxes = aloscene.BoundingBoxes2D(
            pred_boxes, boxes_format="xcyc", absolute=False, device=pred_boxes.device
        )

        target_boxes = torch.cat(
            [
                # Convert to xcyc and relative pos (based on the imge size) ans select
                # into the boxes per batch following the `target_indices` from the Hungarian matching
                boxes2d.xcyc().rel_pos().as_tensor()[indices[b][1]]
                for b, boxes2d in enumerate(frames.boxes2d)
            ],
            dim=0,
        )
        target_boxes = aloscene.BoundingBoxes2D(target_boxes, boxes_format="xcyc", absolute=False)

        # L1 loss
        loss_bbox = F.l1_loss(pred_boxes.as_tensor(), target_boxes.as_tensor(), reduction="none")
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        # Giou loss
        giou = pred_boxes.giou_with(target_boxes)
        loss_giou = 1 - torch.diag(giou)
        losses["loss_giou"] = loss_giou.sum() / num_boxes

        return losses

    def loss_masks(self, outputs: dict, frames: aloscene.Frame, indices: list, num_boxes: torch.Tensor, **kwargs):
        """Compute the losses related to the masks, used sigmoid focal and DICE/F-1 losses

        Parameters
        ----------
        outputs : dict
            Detr model forward outputs
        frames : :mod:`Frames <aloscene.frame>`
            Target frame with boxes2d and labels
        indices : list
            List of tuple with predicted indices and target indices
        num_boxes : torch.Tensor
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

    def _get_src_permutation_idx(self, indices, **kwargs):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_loss(
        self, loss: str, outputs: dict, frames: aloscene.Frame, indices: list, num_boxes: torch.Tensor, **kwargs
    ):
        """Compute a loss given the model outputs, the target frame, the results from the matcher
        and the number of total boxes accross the devices.

        Parameters
        ----------
        loss : str
            Name of the loss to compute
        outputs : dict
            Detr model forward outputs
        frames : :mod:`Frames <aloscene.frame>`
            Trgat frame with boxes2d and labels
        indices : list
            List of tuple with predicted indices and target indices
        num_boxes : torch.Tensor
            Number of total target boxes

        Returns
        -------
        Dict
            Losses of the loss procedure.
        """
        loss_map = {"labels": self.loss_labels, "boxes": self.loss_boxes, "masks": self.loss_masks}
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, frames, indices, num_boxes, **kwargs)

    @torch.no_grad()
    def get_metrics(self, outputs: dict, frames: aloscene.Frame, indices: list, num_boxes: torch.Tensor, **kwargs):
        """Compute some usefull metrics related to the model performance

        Parameters
        ----------
        outputs : dict
            Detr model forward outputs
        frames : :mod:`Frames <aloscene.frame>`
            Trgat frame with boxes2d and labels
        indices : list
            List of tuple with predicted indices and target indices
        num_boxes : torch.Tensor
            Number of total target boxes

        Returns
        -------
        metrics: dict
            - objectness_recall: Percentage of detect object among the GT object (class invariant)
            - recall : Percentage of detect class among all the GT class
            - objectness_true_pos: Among the positive prediction of the model. how much are really positive ?
              (class invariant)
            - precision: Among the positive prediction of the model. how much are well classify ?
            - true_neg:  Among the negative predictions of the model, how much are really negative ?
            - slot_true_neg: Among all the negative slot, how may are predicted as negative ? (class invariant)

        Notes
        -----
        .. important::

            The metrics described above do not reflect directly the true performance of the model.
            The are only directly corredlated with the loss & the hungarian used to train the model. Therefore, the
            computed recall, is NOT the recall, is not the true recall but the recall based on the SLOT & the hungarian
            choice. That being said, it is still a usefull information to monitor the training progress.

        """
        metrics = {}
        if num_boxes == 0:
            return {}

        background_class = len(frames.boxes2d[0].labels.labels_names)

        # Select the target labels in each batch and concat everything
        target_classes_pos = torch.cat(
            [boxes2d.labels.as_tensor()[indices[b][1]] for b, boxes2d in enumerate(frames.boxes2d)]
        )
        target_classes_pos = target_classes_pos.type(torch.long)

        pred_classes = outputs["pred_logits"].argmax(-1)
        idx = self._get_src_permutation_idx(indices)

        target_classes = torch.full(
            pred_classes.shape[:2], background_class, dtype=torch.int64, device=pred_classes.device
        )
        target_classes[idx] = target_classes_pos

        # objectness_recall
        # Percentage of detect object among the GT object (class invariant)
        gt_objects_filter = target_classes != background_class
        gt_pred_clases = pred_classes[gt_objects_filter]
        len_gt_objects_filter = len(target_classes[gt_objects_filter])
        len_gt_pred_clases = len(gt_pred_clases[gt_pred_clases != background_class])
        if len_gt_objects_filter != 0 and len_gt_pred_clases != 0:
            metrics["objectness_recall"] = len_gt_pred_clases / len_gt_objects_filter
        elif len_gt_pred_clases > 0:
            metrics["objectness_recall"] = 0

        # Recall
        # percentage of detect class among all the GT class
        metrics["recall"] = (gt_pred_clases == target_classes[gt_objects_filter]).to(torch.float16).mean()

        # objectness_true_pos
        # Among the positive prediction of the model. how much are really positive ? (class invariant)
        pred_pos_filter = pred_classes != background_class
        pred_gt_classes = target_classes[pred_pos_filter]
        len_pred_pos_filter = len(pred_classes[pred_pos_filter])
        len_pred_gt_classes = len(pred_gt_classes[pred_gt_classes != background_class])
        if len_pred_pos_filter != 0 and len_pred_gt_classes != 0:
            metrics["objectness_true_pos"] = len_pred_gt_classes / len_pred_pos_filter
        elif len_pred_gt_classes > 0:
            metrics["objectness_true_pos"] = 0

        # precision
        # Among the positive prediction of the model. how much are well classify ?
        if len(pred_gt_classes) > 0:
            metrics["precision"] = (pred_gt_classes == pred_classes[pred_pos_filter]).to(torch.float16).mean()

        # true_neg
        # Among the negative predictions of the model, how much are really negative ? (class invariant)
        pred_neg_filter = pred_classes == background_class
        pred_gt_classes = target_classes[pred_neg_filter]
        len_pred_gt_classes = len(pred_gt_classes[pred_gt_classes == background_class])
        len_pred_classes = len(pred_classes[pred_neg_filter])
        if len_pred_classes != 0 and len_pred_gt_classes != 0:
            metrics["true_neg"] = len_pred_gt_classes / len_pred_classes
        elif len_pred_gt_classes != 0:
            metrics["true_neg"] = 0

        # slot_true_neg
        # Among all the negative slot, how may are predicted as negative ? (class invariant)
        gt_neg_filter = target_classes == background_class
        gt_pred_clases = pred_classes[gt_neg_filter]
        len_gt_pred_clases = len(gt_pred_clases[gt_pred_clases == background_class])
        len_target_classes = len(target_classes[gt_neg_filter])
        if len_gt_pred_clases != 0 and len_target_classes != 0:
            metrics["slot_true_neg"] = len_gt_pred_clases / len_target_classes
        elif len_gt_pred_clases != 0:
            metrics["slot_true_neg"] = 0

        return metrics

    @torch.no_grad()
    def get_statistical_metrics(
        self, outputs: dict, frames: aloscene.Frame, indices: list, num_boxes: torch.Tensor, **kwargs
    ):
        """Compute some usefull statistical metrics about the model outputs and the inputs.

        Parameters
        ----------
        outputs : dict
            Detr model forward outputs
        frames : :mod:`Frames <aloscene.frame>`
            Target frame with boxes2d and labels
        indices : list
            List of tuple with predicted indices and target indices
        num_boxes : torch.Tensor
            Number of total target boxes

        Returns
        -------
        dict
            Metrics
        """
        if num_boxes == 0:
            return {}

        background_class = len(frames.boxes2d[0].labels.labels_names)

        # Select the target labels in each batch and concat everything
        target_classes_pos = torch.cat(
            [boxes2d.labels.as_tensor()[indices[b][1]] for b, boxes2d in enumerate(frames.boxes2d)]
        )
        target_classes_pos = target_classes_pos.type(torch.long)
        pred_classes = outputs["pred_logits"].argmax(-1)
        idx = self._get_src_permutation_idx(indices)
        target_classes = torch.full(
            pred_classes.shape[:2], background_class, dtype=torch.int64, device=pred_classes.device
        )
        target_classes[idx] = target_classes_pos

        # Predicted class and target class histogram
        histogram_p_class = pred_classes[pred_classes != background_class].flatten()
        histogram_t_class = target_classes[target_classes != background_class].flatten()

        # Scatter predicted objects size & Objects position
        pred_boxes = outputs["pred_boxes"]
        target_boxes = torch.cat(
            [boxes2d.xcyc().rel_pos().as_tensor() for b, boxes2d in enumerate(frames.boxes2d)], dim=0
        )

        pred_boxes = pred_boxes[pred_classes != background_class]
        scatter_p_boxes_size = pred_boxes[:, 2:]
        scatter_p_boxes_pos = torch.cat([pred_boxes[:, 0:1], 1 - pred_boxes[:, 1:2]], dim=-1)
        scatter_t_boxes_size = target_boxes[:, 2:]
        scatter_t_boxes_pos = torch.cat([target_boxes[:, 0:1], 1 - target_boxes[:, 1:2]], dim=-1)

        return {
            "histogram_p_class": histogram_p_class.to(torch.float32),
            "histogram_t_class": histogram_t_class.to(torch.float32),
            "scatter_p_boxes_size": (["width", "height"], scatter_p_boxes_size),
            "scatter_p_boxes_pos": (["x", "y"], scatter_p_boxes_pos),
            "scatter_t_boxes_size": (["width", "height"], scatter_t_boxes_size),
            "scatter_t_boxes_pos": (["x", "y"], scatter_t_boxes_pos),
        }

    def forward(
        self,
        m_outputs: dict,
        frames: aloscene.Frame,
        matcher_frames: aloscene.Frame = None,
        compute_statistical_metrics: bool = False,
        **kwargs,
    ):
        """This performs the loss computation.

        Parameters
        ----------
        outputs: dict
            Dict of tensors, see the output specification of the model for the format
        targets: Frame
            Target frames
        compute_statistical_metrics: bool
            Whether to compute statistical data bout the model outputs/inputs, by default False

        Returns
        -------
        torch.tensor
            Total loss as weighting of losses
        dict
            Individual losses
        """
        assert isinstance(frames, aloscene.Frame)
        assert isinstance(frames.boxes2d[0], aloscene.BoundingBoxes2D)
        assert frames.boxes2d[0].labels is not None and frames.boxes2d[0].labels.encoding == "id"
        matcher_frames = matcher_frames if matcher_frames is not None else frames

        outputs_without_aux = {k: v for k, v in m_outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, matcher_frames, **kwargs)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(boxes2d.labels.shape[0] for boxes2d in frames.boxes2d)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(m_outputs.values())).device)

        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, m_outputs, frames, indices, num_boxes))

        metrics = self.get_metrics(m_outputs, frames, indices, num_boxes)
        if compute_statistical_metrics:
            metrics.update(self.get_statistical_metrics(m_outputs, frames, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in m_outputs:
            for i, aux_outputs in enumerate(m_outputs["aux_outputs"]):

                indices = self.matcher(aux_outputs, matcher_frames, **kwargs)

                for loss in self.losses:
                    if loss == "masks":
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    if loss == "labels":
                        # Logging is enabled only for the last layer
                        kwargs.update({"log": False})
                    l_dict = self.get_loss(loss, aux_outputs, frames, indices, num_boxes, **kwargs)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}

                    losses.update(l_dict)

        total_loss = sum(losses[k] * self.loss_weights[k] for k in losses.keys())
        # TODO reduce losses over all GPUs for logging purposes
        losses.update(metrics)

        return total_loss, losses
