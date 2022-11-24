# Mostly inspired from https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/deformable_detr.py#L198

import torch

import torch.nn.functional as F
import aloscene
from alonet.detr import DetrCriterion


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_boxes: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    weight: torch.Tensor = None,
) -> torch.Tensor:
    """Sigmoid focal loss for classification

    Parameters
    ----------
    inputs : torch.Tensor
        The predictions for each example.
    targets : torch.Tensor
        A float tensor with the same shape as inputs. Stores the binary
        classification label for each element in inputs
        (0 for the negative class and 1 for the positive class).
    num_boxes : torch.Tensor
    alpha : float, optional
        (optional) Weighting factor in range (0,1) to balance
        positive vs negative examples. -1 for no weighting. By default 0.25.
    gamma : float, optional
        Exponent of the modulating factor (1 - p_t) to
        balance easy vs hard examples. By default 2

    Returns
    -------
    torch.Tensor
        Scalar
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, weight, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean()


class DeformableCriterion(DetrCriterion):
    """This class computes the loss for Deformable DETR. The process happens in two steps

        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)

    This Criterion is rouhgly smilar to `DetrCriterion` except for the labels loss and the metrics.
    """

    def __init__(self, loss_label_weight: float, focal_alpha=0.25, **kwargs):
        """Create the criterion.

        Parameters
        ----------
        loss_label_weight : float
            Label class weight, to use in CE or sigmoid focal (default) loss (depends of network configuration)
        focal_alpha : float, optional
            This parameter is used only when the model use sigmoid activation function.
            Weighting factor in range (0,1) to balance positive vs negative examples. -1 for no weighting,
            by default 0.25

        Notes
        -----
        This criterion detect automatically deformable activation function to compute the loss.
            * If forward_out["activate_fn"] == "softmax", cross entropy loss will be computed
            * If forward_out["activate_fn"] == "sigmoid", sigmoid focal loss will be computed
        """

        if "loss_ce_weight" in kwargs or "loss_focal_label" in kwargs:
            raise Exception(
                f"Usage not supported in class {self.__class__.__name__}. The weight of ce or focal for labels losses"
                + "must be provided on the 'loss_label_weight' attribute."
            )

        kwargs["loss_ce_weight"] = loss_label_weight  # Add loss_label_weight as loss_ce_weight and loss_focal_label
        super().__init__(**kwargs)
        self.focal_alpha = focal_alpha
        self.loss_weights["loss_focal_label"] = loss_label_weight

        if kwargs["aux_loss_stage"] > 0:
            for i in range(kwargs["aux_loss_stage"] - 1):
                self.loss_weights.update({f"loss_focal_label_{i}": loss_label_weight})

    def loss_labels(
        self, outputs: dict, frames: aloscene.Frame, indices: list, num_boxes: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Compute the classification loss

        Parameters
        ----------
        outputs : dict
            model forward outputs
        frames : aloscene.Frame
            Target frame with ground truth boxes2d and labels
        indices : list
            List of tuple with matching predicted indices and target indices. `len(indices)` is equal to batch size.
        num_boxes : torch.Tensor
            Number of total target boxes

        Returns
        -------
        torch.Tensor
            Classification loss
        """
        if "activation_fn" not in outputs:
            raise Exception("'activation_fn' must be declared in forward output.")
        if outputs["activation_fn"] == "softmax":
            return super().loss_labels(outputs, frames, indices, num_boxes, **kwargs)

        assert frames.names[0] == "B"
        assert frames.boxes2d[0].labels is not None and frames.boxes2d[0].labels.encoding == "id"

        num_classes = len(frames.boxes2d[0].labels.labels_names)

        pred_logits = outputs["pred_logits"]  # (b, nb_slots, nb_classes)
        idx = self._get_src_permutation_idx(indices)
        # Select the target labels in each batch and concat everything
        target_classes_pos = torch.cat(
            [boxes2d.labels.as_tensor()[indices[b][1]] for b, boxes2d in enumerate(frames.boxes2d)]
        )
        target_classes_pos = target_classes_pos.type(torch.long)
        # target_classes (b, nb_slots)
        # positive slot is assigned with class id
        # negative slot is assigned with a virtual background class whose id is equal num_classes
        target_classes = torch.full(
            pred_logits.shape[:2], num_classes, dtype=torch.int64, device=pred_logits.device
        )  # (b, nb_slots)
        target_classes[idx] = target_classes_pos

        # target_classes_onehot (b, nb_slots, nb_classes + 1)
        # + 1 because we have an additional background class
        target_classes_onehot = torch.zeros(
            [pred_logits.shape[0], pred_logits.shape[1], pred_logits.shape[2] + 1],
            dtype=pred_logits.dtype,
            layout=pred_logits.layout,
            device=pred_logits.device,
        )
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        # remove "phantom" class from onehot
        # target_classes_onehot (b, nb_slots, nb_classes)
        target_classes_onehot = target_classes_onehot[:, :, :-1]

        if frames.labels is not None and "traffic_lights_annotated" in frames.labels:
            # multiply traffic_light class by zero when not available in labels (removes gradient)
            weight = torch.ones_like(pred_logits)
            weight[:, :, -1] = weight[:, :, -1] * frames.labels["traffic_lights_annotated"].as_tensor().unsqueeze(-1)
        else:
            weight = None

        loss_focal = (
            sigmoid_focal_loss(
                pred_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2, weight=weight
            )
            * pred_logits.shape[1]
        )
        losses = {"loss_focal_label": loss_focal}  # Compute loss_focal_label instead of loss_ce

        return losses

    @torch.no_grad()
    def get_metrics(self, outputs: dict, frames: aloscene.Frame, indices: list, num_boxes: torch.Tensor) -> dict:
        """Compute some usefull metrics related to the model performance

        Parameters
        ----------
        outputs : dict
            model forward outputs
        frames : aloscene.Frame
            Target frame with ground truth boxes2d and labels
        indices : list
            List of tuple with matching predicted indices and target indices. `len(indices)` is equal to batch size.
        num_boxes : torch.Tensor
            Number of total target boxes

        Returns
        -------
        dict
            - "recall" : Percentage of detected class among all the GT class
            - "precision": Among the positive prediction of the model. How much are well classify ?

        Notes
        ------
        Metrics are calculated based on threshold = 0.3 to filter out positive prediction.
        Positive prediction = prediction with score >= threshold, otherwise negative prediction
        This threshold value can be change in inference. That being said, it is still a usefull
        information to monitor the training progress.
        """
        if "activation_fn" not in outputs:
            raise Exception("'activation_fn' must be declared in forward output.")
        if outputs["activation_fn"] == "softmax":
            return super().get_metrics(outputs, frames, indices, num_boxes)

        threshold = 0.3
        metrics = {}
        if num_boxes == 0:
            return {}

        # virtual background class
        background_class = len(frames.boxes2d[0].labels.labels_names)

        # Select the target labels in each batch and concat everything
        target_classes_pos = torch.cat(
            [boxes2d.labels.as_tensor()[indices[b][1]] for b, boxes2d in enumerate(frames.boxes2d)]
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
