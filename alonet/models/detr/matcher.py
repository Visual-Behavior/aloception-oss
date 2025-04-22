# Mostly copy past from https://github.com/facebookresearch/detr

""" Modules to compute the matching cost and solve the corresponding LSAP.
"""

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

# from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou

import aloscene


class DetrHungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_boxes: float = 1, cost_giou: float = 1):
        """Creates the matcher

        Parameters
        ----------
        cost_class: (float)
            This is the relative weight of the classification error in the matching cost
        cost_boxes: (float)
            This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
        cost_giou: (float)
            This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_boxes = cost_boxes
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_boxes != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def hungarian_cost_class(self, tgt_boxes: aloscene.BoundingBoxes2D, m_outputs: dict, **kwargs):
        """
        Compute the cost class for the Hungarina matcher

        Parameters
        ----------
        m_outputs: dict
            Dict output of the alonet.detr.Detr model. This is a dict that contains at least these entries:
            "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
            "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

        tgt_boxes: aloscene.BoundingBoxes2D
            Target boxes2d across the batch
        """
        out_prob = m_outputs["pred_logits"]
        out_prob = out_prob.flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]

        # Retrieve the target ID for each boxes 2d
        tgt_ids = tgt_boxes.labels.type(torch.long).rename_(None)
        assert len(tgt_ids.shape) == 1

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids].as_tensor()

        return cost_class

    @torch.no_grad()
    def hungarian_cost_l1_boxes(self, tgt_boxes: aloscene.BoundingBoxes2D, m_outputs: dict, **kwargs):
        """
        Compute l1 cost between boxes

        Parameters
        ----------
        m_outputs: dict
            Dict output of the alonet.detr.Detr model. This is a dict that contains at least these entries:
            "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
            "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

        tgt_boxes: aloscene.BoundingBoxes2D
            Target boxes2d across the batch
        """
        out_bbox = m_outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
        # out_bbox = aloscene.BoundingBoxes2D(out_bbox, absolute=False, boxes_format="xcyc", names=("N", None))
        assert tgt_boxes.boxes_format == "xcyc" and not tgt_boxes.absolute
        tgt_boxes = tgt_boxes.as_tensor()

        # assert out_bbox.boxes_format == "xcyc" and not out_bbox.absolute
        cost_boxes = torch.cdist(out_bbox, tgt_boxes, p=1)

        return cost_boxes

    @torch.no_grad()
    def hungarian_cost_giou_boxes(self, tgt_boxes: aloscene.BoundingBoxes2D, m_outputs: dict, **kwargs):
        """
        Compute GIOU cost between boxes

        Parameters
        ----------
        m_outputs: dict
            Dict output of the alonet.detr.Detr model. This is a dict that contains at least these entries:
            "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
            "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

        tgt_boxes: aloscene.BoundingBoxes2D
            Target boxes2d across the batch
        """
        out_bbox = m_outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
        out_bbox = aloscene.BoundingBoxes2D(out_bbox, absolute=False, boxes_format="xcyc", names=("N", None))
        assert tgt_boxes.boxes_format == "xcyc" and not tgt_boxes.absolute
        try:
            cost_giou = -out_bbox.giou_with(tgt_boxes)
        except Exception as e:
            print('m_outputs["pred_boxes"]', m_outputs["pred_boxes"])
            raise e

        return cost_giou

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
            "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
        frames: aloscene.Frame
            Target frame with a set of boxes2d named : "gt_boxes_2d" with labels.

        Returns
        -------
            A list of size batch_size, containing tuples of (index_i, index_j) where:
            - index_i is the indices of the selected predictions (in order)
            - index_j is the indices of the corresponding selected targets (in order)

            For each batch element, it holds:
            len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        assert isinstance(frames, aloscene.Frame)
        assert isinstance(frames.boxes2d[0], aloscene.BoundingBoxes2D)
        assert frames.boxes2d[0].labels is not None and frames.boxes2d[0].labels.encoding == "id"

        bs, num_queries = m_outputs["pred_logits"].shape[:2]

        tgt_boxes = torch.cat([boxes.rel_pos().xcyc().remove_padding() for boxes in frames.boxes2d], dim=0)

        # No GT boxes
        if tgt_boxes.shape[0] == 0:
            return [
                (torch.as_tensor([], dtype=torch.int64), torch.as_tensor([], dtype=torch.int64)) for b in range(0, bs)
            ]

        # Class cost
        cost_class = self.hungarian_cost_class(tgt_boxes, m_outputs, **kwargs)
        # Compute the L1 cost between boxes
        cost_boxes = self.hungarian_cost_l1_boxes(tgt_boxes, m_outputs, **kwargs)
        # Compute the Giou cost
        cost_giou = self.hungarian_cost_giou_boxes(tgt_boxes, m_outputs, **kwargs)

        # Final cost matrix
        C = self.cost_boxes * cost_boxes + self.cost_class * cost_class + self.cost_giou * cost_giou

        # (batch, num_queries, total_targets)
        C = C.view(bs, num_queries, -1).cpu()

        # Retrieve the number of target per batch
        sizes = [boxes.labels.shape[0] for boxes in frames.boxes2d]

        # Retrieve the p_indices & t_indices for each batch
        batch_cost_matrix = [c[i] for i, c in enumerate(C.split(sizes, -1))]

        return self.hungarian(batch_cost_matrix, **kwargs)


def build_matcher(args):
    return DetrHungarianMatcher(
        cost_class=args.set_cost_class, cost_boxes=args.set_cost_boxes, cost_giou=args.set_cost_giou
    )
