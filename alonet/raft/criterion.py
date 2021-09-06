from torch import nn
import torch

import aloscene
from aloscene import Flow


class RAFTCriterion(nn.Module):
    def __init__(self):
        super().__init__()

    # loss from RAFT implementation
    @staticmethod
    def sequence_loss(flow_preds, flow_gt, valid=None, gamma=0.8, max_flow=400, compute_per_iter=False):
        """Loss function defined over sequence of flow predictions"""
        n_predictions = len(flow_preds)
        flow_loss = 0.0

        # exlude invalid pixels and extremely large diplacements
        mag = torch.sum(flow_gt ** 2, dim=1).sqrt()
        if valid is None:
            valid = torch.ones_like(mag, dtype=torch.bool)
        else:
            valid = (valid >= 0.5) & (mag < max_flow)
        for i in range(n_predictions):
            i_weight = gamma ** (n_predictions - i - 1)
            i_loss = (flow_preds[i] - flow_gt).abs()
            flow_loss += i_weight * (valid[:, None] * i_loss).mean()

        if compute_per_iter:
            epe_per_iter = []
            for flow_p in flow_preds:
                epe = torch.sum((flow_p - flow_gt) ** 2, dim=1).sqrt()
                epe = epe.view(-1)[valid.view(-1)]
                epe_per_iter.append(epe)
        else:
            epe_per_iter = None

        epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()
        epe = epe.view(-1)[valid.view(-1)]

        metrics = {
            "loss": flow_loss.item(),
            "epe": epe.mean().item(),
            "1px": (epe < 1).float().mean().item(),
            "3px": (epe < 3).float().mean().item(),
            "5px": (epe < 5).float().mean().item(),
        }
        return flow_loss, metrics, epe_per_iter

    def forward(self, m_outputs, frame1, use_valid=True, compute_per_iter=False):
        assert isinstance(frame1, aloscene.Frame)
        flow_preds = m_outputs
        flow_gt = [f.batch() for f in frame1.flow["flow_forward"]]
        flow_gt = torch.cat(flow_gt, dim=0)
        # occlusion mask -- not used in raft original repo
        # in raft, valid removes only pixels with ground_truth flow > 1000 on one dimension
        # valid = (flow_gt.occlusion / 255.)
        # valid = valid[valid.get_slices({"C":0})]
        assert flow_gt.names == ("B", "C", "H", "W")
        flow_gt = flow_gt.as_tensor()
        flow_x, flow_y = flow_gt[:, 0, ...], flow_gt[:, 1, ...]
        if use_valid:
            valid = (flow_x.abs() < 1000) & (flow_y.abs() < 1000)
        else:
            valid = None
        flow_loss, metrics, epe_per_iter = RAFTCriterion.sequence_loss(
            flow_preds, flow_gt, valid, compute_per_iter=compute_per_iter
        )
        return flow_loss, metrics, epe_per_iter
