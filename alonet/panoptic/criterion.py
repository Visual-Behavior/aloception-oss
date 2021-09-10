import torch
from torch import nn

from alonet.multi_gpu import get_world_size, is_dist_avail_and_initialized
import torch.nn.functional as F
import aloscene


def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
            balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


class PanopticCriterion(nn.Module):
    """This class computes the loss for Panoptic.
    Each pair of matched ground-truth / prediction (supervise class and box)"""

    def __init__(
        self, matcher: nn.Module, loss_dice_weight: float, loss_focal_weight: float, aux_loss_stage: int, losses,
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
        """
        super().__init__()
        self.matcher = matcher
        self.losses = losses

        # Define the weight dict
        loss_weights = {"loss_DICE": loss_dice_weight, "loss_focal": loss_focal_weight}
        if aux_loss_stage > 0:
            aux_loss_weights = {}
            for i in range(aux_loss_stage - 1):
                aux_loss_weights.update({k + f"_{i}": v for k, v in loss_weights.items()})
            loss_weights.update(aux_loss_weights)
        self.loss_weights = loss_weights

    def _get_src_permutation_idx(self, indices, **kwargs):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def loss_masks(self, outputs: dict, frames: aloscene.Frame, indices: list, num_boxes: torch.Tensor, **kwargs):
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
        """

        losses = {}
        if num_boxes == 0:
            return {}

        idx = self._get_src_permutation_idx(indices)
        target_masks = torch.cat(
            [
                # Select masks per batch following the `target_indices` from the Hungarian matching
                masks.as_tensor()[indices[b][1]]
                for b, masks in enumerate(frames.segmentation)
            ],
            dim=0,
        )

        pred_masks = outputs["pred_masks"][idx]
        pred_masks = F.interpolate(pred_masks[:, None], size=target_masks.shape[-2:], mode="bilinear", align_corners=False)
        pred_masks = pred_masks[:,0].flatten(1)
        target_masks = target_masks.flatten(1).view(pred_masks.shape)

        # DICE/F-1 loss
        losses["loss_DICE"] = dice_loss(pred_masks, target_masks, num_boxes)

        # Sigmoid focal loss
        losses["loss_focal"] = sigmoid_focal_loss(pred_masks, target_masks, num_boxes, **kwargs)

        return losses

    @torch.no_grad()
    def get_metrics(self, outputs: dict, frames: aloscene.Frame, indices: list, num_boxes: torch.Tensor, **kwargs):
        """Compute some usefull metrics related to the model performance

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
        metrics: dict
            - "recall" : Percentage of detected class among all the GT class
            - "precision": Among the positive prediction of the model. How much are well classify ?

        One important thing: The metrics described above do not reflect directly the true performance of the model.
        The are only directly corredlated with the loss & the hungarian used to train the model. Therefore, the
        computed recall, is NOT the recall, is not the true recall but the recall based on the SLOT & the hungarian
        choice. That being said, it is still a usefull information to monitor the training progress.
        """

        threshold = 0.3
        metrics = {}
        if num_boxes == 0:
            return {}

        # virtual background class
        background_class = len(frames.segmentation[0].labels.labels_names)

        # Select the target labels in each batch and concat everything
        target_classes_pos = torch.cat(
            [masks.labels.as_tensor()[indices[b][1]] for b, masks in enumerate(frames.segmentation)]
        )
        target_classes_pos = target_classes_pos.type(torch.long)

        # pred_classes (b, nb_slots)
        outs_probs = outputs["pred_logits"].sigmoid()
        pred_scores, pred_classes = outs_probs.max(-1)
        idx = self._get_src_permutation_idx(indices)

        # target_clases (b, nb_slots)
        target_classes = torch.full(
            pred_classes.shape[:2], background_class, dtype=torch.int64, device=pred_classes.device
        )
        target_classes[idx] = target_classes_pos

        # Number of true positive prediction
        true_positive_pred = (pred_classes == target_classes)[pred_scores >= threshold].sum()
        positive_prediction = (pred_scores >= threshold).sum()

        metrics["precision"] = (
            true_positive_pred / positive_prediction
            if positive_prediction > 0
            else torch.zeros(()).to(pred_classes.device)
        )
        metrics["recall"] = (
            true_positive_pred / len(target_classes_pos)
            if len(target_classes_pos) > 0
            else torch.zeros(()).to(pred_classes.device)
        )

        return metrics

    @torch.no_grad()
    def get_statistical_metrics(
        self, outputs: dict, frames: aloscene.Frame, indices: list, num_boxes: torch.Tensor, **kwargs
    ):
        """Compute some usefull statistical metrics about the model outputs and the inputs.

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
        metrics: dict
        """
        if num_boxes == 0:
            return {}

        background_class = len(frames.segmentation[0].labels.labels_names)

        # Select the target labels in each batch and concat everything
        target_classes_pos = torch.cat(
            [masks.labels.as_tensor()[indices[b][1]] for b, masks in enumerate(frames.segmentation)]
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

        return {
            "histogram_p_class": histogram_p_class.to(torch.float32),
            "histogram_t_class": histogram_t_class.to(torch.float32),
        }

    def get_loss(
        self, loss: str, outputs: dict, frames: aloscene.Frame, indices: list, num_boxes: torch.Tensor, **kwargs
    ):
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
        """
        loss_map = {"masks": self.loss_masks}
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, frames, indices, num_boxes, **kwargs)

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
        targets: aloscene.Frame
            ...
        compute_statistical_metrics: bool
            Whether to compute statistical data bout the model outputs/inputs. (False by default)
        """
        assert isinstance(frames, aloscene.Frame)
        assert isinstance(frames.segmentation[0], aloscene.Mask)
        assert frames.segmentation[0].labels is not None and frames.segmentation[0].labels.encoding == "id"
        matcher_frames = matcher_frames if matcher_frames is not None else frames

        outputs_without_aux = {k: v for k, v in m_outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, matcher_frames, **kwargs)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(masks.labels.shape[0] for masks in frames.segmentation)
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
