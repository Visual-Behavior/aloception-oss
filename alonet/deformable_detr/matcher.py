import torch
import aloscene
from alonet.detr.matcher import DetrHungarianMatcher


class DeformableDetrHungarianMatcher(DetrHungarianMatcher):
    @torch.no_grad()
    def hungarian_cost_class(self, tgt_boxes: aloscene.BoundingBoxes2D, m_outputs: dict, **kwargs):
        """
        Compute the cost class for the Hungarian matcher

        Parameters
        ----------
        m_outputs: dict
            Dict output of the alonet.detr.Detr model. This is a dict that contains at least these entries:
                - "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                - "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

        tgt_boxes: aloscene.BoundingBoxes2D
            Target boxes2d across the batch
        """
        out_prob = m_outputs["pred_logits"]
        # Retrieve the target ID for each boxes 2d
        tgt_ids = tgt_boxes.labels.type(torch.long).rename_(None)  # [total number of target boxes in batch,]

        if "activation_fn" not in m_outputs:
            raise Exception("'activation_fn' must be declared in forward output.")
        if m_outputs["activation_fn"] == "softmax":
            out_prob = out_prob.flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -out_prob[:, tgt_ids].as_tensor()  # [batch_size * num_queries, total nb of targets in batch]
        else:
            out_prob = out_prob.flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]
            alpha = 0.25
            gamma = 2.0
            neg_cost_class = (1 - alpha) * (out_prob**gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
            cost_class = cost_class.as_tensor()
        return cost_class
