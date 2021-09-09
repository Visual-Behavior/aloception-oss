# Inspired by the official DETR repository and adapted for aloception
"""
Panoptic module to predict object segmentation.
"""
import torch
import torch.nn.functional as F
from torch import nn

from alonet.detr.detr_r50 import DetrR50
from alonet.panoptic.nn import FPNstyleCNN
from alonet.panoptic.nn import MHAttentionMap

import alonet
import aloscene


class Panoptic(nn.Module):
    def __init__(
        self, DETR_module, freeze_detr=False, device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.detr = DETR_module
        self.detr.return_dec_outputs = True  # Get decode outputs in forward

        if freeze_detr:
            for p in self.parameters():
                p.requires_grad_(False)

        hidden_dim, nheads = DETR_module.transformer.d_model, DETR_module.transformer.nhead
        self.bbox_attention = MHAttentionMap(hidden_dim, hidden_dim, nheads, dropout=0.0)
        self.mask_head = FPNstyleCNN(hidden_dim + nheads, [1024, 512, 256], hidden_dim)

        if device is not None:
            self.to(device)

        self.device = device

    def forward(self, frames: aloscene.frame, **kwargs):
        # DETR model forward to obtain box embeddings
        out = self.detr(frames, **kwargs)
        features = out["dec_features"]
        src = features[-1][0]
        bs = src.shape[0]

        # FIXME h_boxes takes the last one computed, keep this in mind
        # Use box embeddings as input of Multi Head attention
        bbox_mask = self.bbox_attention(out["dec_outputs"][-1], out["dec_outputs_mem"], mask=out["dec_mask"])
        # And then, use MHA ouput as input of FPN-style CNN
        seg_masks = self.mask_head(self.detr.input_proj(src), bbox_mask, [features[i][0] for i in range(3)[::-1]])
        out["pred_masks"] = seg_masks.view(bs, self.detr.num_queries, seg_masks.shape[-2], seg_masks.shape[-1])
        return out  # Return the DETR output + pred_masks


if __name__ == "__main__":
    device = torch.device("cuda")

    # Load model
    model = DetrR50(num_classes=91, weights="detr-r50", device=device).eval()
    model = Panoptic(model)
    model.to(device)

    # Random frame
    frame = aloscene.Frame(torch.rand((3, 250, 250)), names=("C", "H", "W")).norm_resnet().to(device)
    frame = frame.batch_list([frame, frame, frame])

    # Pred of size (B, NQ, H//4, W//4)
    print(model(frame)["pred_masks"].shape)
