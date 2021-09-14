# Mostly copy past from https://github.com/facebookresearch/detr

""" Modules to compute the matching cost and solve the corresponding LSAP.
"""

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

# from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou

import aloscene


class PanopticHungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    @torch.no_grad()
    def hungarian_cost_class(self, tgt_labels: aloscene.Labels, m_outputs: dict, **kwargs):
        """
        Compute the cost class for the Hungarina matcher

        Parameters
        ----------
        m_outputs: dict
            Dict output of the alonet.detr.Detr model. This is a dict that contains at least these entries:
            "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits

        tgt_labels: aloscene.Labels
            Target labels across the batch
        """
        out_prob = m_outputs["pred_logits"]
        out_prob = out_prob.flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]

        # Retrieve the target ID for each boxes 2d
        tgt_ids = tgt_labels.type(torch.long).rename_(None)
        assert len(tgt_ids.shape) == 1

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids].as_tensor()

        return cost_class

    def hungarian(self, batch_cost_matrix: list, **kwargs):
        # Retrieve the p_indices & t_indices for each batch
        indices = [linear_sum_assignment(c) for c in batch_cost_matrix]
        final_indices = [
            (torch.as_tensor(p_indices, dtype=torch.int64), torch.as_tensor(t_indices, dtype=torch.int64))
            for p_indices, t_indices in indices
        ]
        return final_indices

    @torch.no_grad()
    def forward(self, m_outputs: dict, frames: aloscene.Frame, **kwargs):
        """Performs the matching

        Parameters
        ----------
        m_outputs: dict
            Dict output of the alonet.detr.Detr model. This is a dict that contains at least these entries:
            "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
        frames: aloscene.Frame
            Target frame with a set of segmentation named : "gt_segmentation" with labels.

        Returns
        -------
            A list of size batch_size, containing tuples of (index_i, index_j) where:
            - index_i is the indices of the selected predictions (in order)
            - index_j is the indices of the corresponding selected targets (in order)

            For each batch element, it holds:
            len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        assert isinstance(frames, aloscene.Frame)
        assert isinstance(frames.segmentation[0], aloscene.Mask)
        assert frames.segmentation[0].labels is not None and frames.segmentation[0].labels.encoding == "id"

        bs, num_queries = m_outputs["pred_logits"].shape[:2]
        tgt_labels = torch.cat([segm.labels for segm in frames.segmentation], dim=0)

        # No GT labels
        if tgt_labels.shape[0] == 0:
            return [
                (torch.as_tensor([], dtype=torch.int64), torch.as_tensor([], dtype=torch.int64)) for b in range(0, bs)
            ]

        # Class cost
        cost_class = self.hungarian_cost_class(tgt_labels, m_outputs, **kwargs)
        cost_class = cost_class.view(bs, num_queries, -1).cpu()  # (batch, num_queries, total_targets)

        # Retrieve the number of target per batch
        sizes = [segm.labels.shape[0] for segm in frames.segmentation]

        # Retrieve the p_indices & t_indices for each batch
        batch_cost_matrix = [c[i] for i, c in enumerate(cost_class.split(sizes, -1))]

        return self.hungarian(batch_cost_matrix, **kwargs)


def build_matcher(args):
    """Matcher by default"""
    return PanopticHungarianMatcher()
