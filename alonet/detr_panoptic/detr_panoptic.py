# Inspired by the official DETR repository and adapted for aloception
# https://github.com/facebookresearch/detr/blob/eb9f7e03ed8e2ed2cd55528989fe7df890bc3fc0/models/detr.py
"""
Panoptic module to use in object detection/segmentation tasks.
"""
from typing import Callable, Dict, Union
from collections import namedtuple
import argparse
import time

import torch
import torch.nn.functional as F
from torch import nn

from alonet.detr_panoptic.nn import FPNstyleCNN
from alonet.detr_panoptic.nn import MHAttentionMap
from alonet.detr_panoptic.utils import get_mask_queries
from alonet.detr_panoptic.misc import assert_and_export_onnx

import aloscene
import alonet


class PanopticHead(nn.Module):
    """Pytorch head module to predict segmentation masks from previous boxes detection task.

    Parameters
    ----------
    DETR_module : :mod:`alonet.detr.detr`
        Object detection module based on :mod:`DETR <alonet.detr.detr>` architecture
    freeze_detr : bool, optional
        Freeze :attr:`DETR_module` weights in train procedure, by default True
    aux_loss : bool, optional
        Return aux outputs in forward step (if required), by default use :attr:`DETR_module.aux_loss` attribute value
    return_pred_outputs : bool, optional
        Return attr:`pred_logits` and attr:`pred_boxes` (detr outputs), by default True
    return_detr_outputs : bool, optional
        Return complementary detr outputs when calling :func: `forward`, by default False
    device : torch.device, optional
        Configure module in CPU or GPU, by default :attr:`torch.device("cpu")`
    weights : str, optional
        Load weights from name project, by default None
    fpn_list : list, optional
        Expected features backbone sizes from [layer1, layer2, layer3], by default [1024, 512, 256]
    strict_load_weights : bool
        Load the weights (if any given) with strict = ``True`` (by default)
    tracing : bool, Optional
        Change model behavior to be exported as TorchScript, by default False
    """

    INPUT_MEAN_STD = alonet.detr.detr.INPUT_MEAN_STD

    def __init__(
        self,
        DETR_module: alonet.detr.Detr,
        freeze_detr: bool = True,
        aux_loss: bool = None,
        return_pred_outputs: bool = True,
        return_detr_outputs: bool = False,
        device: torch.device = torch.device("cpu"),
        weights: str = None,
        fpn_list: list = [1024, 512, 256],
        strict_load_weights: bool = True,
        tracing: bool = False,
    ):
        super().__init__()
        self.detr = DETR_module

        # Get complement outputs in forward
        self.detr.return_dec_outputs = True
        self.detr.return_enc_outputs = True
        self.detr.return_bb_outputs = True
        self.tracing = tracing
        self.return_detr_outputs = return_detr_outputs
        self.return_pred_outputs = return_pred_outputs
        self.detr.aux_loss = aux_loss if aux_loss is not None else self.detr.aux_loss

        if self.tracing and self.detr.aux_loss:
            raise AttributeError("When tracing = True, aux_loss must be False")

        # Freeze DETR parameters to not train them
        if freeze_detr:
            for p in self.parameters():
                p.requires_grad_(False)

        # Use values provides of DETR module in MHA and FPN
        hidden_dim, nheads = DETR_module.transformer.d_model, DETR_module.transformer.nhead
        self.bbox_attention = MHAttentionMap(hidden_dim, hidden_dim, nheads, dropout=0.1)
        self.mask_head = FPNstyleCNN(hidden_dim + nheads, fpn_list, hidden_dim)

        if device is not None:
            self.to(device)

        self.device = device

        # Load weights
        if weights is not None:
            if ".pth" in weights or ".ckpt" in weights or weights in ["detr-r50-panoptic"]:
                alonet.common.load_weights(self, weights, device, strict_load_weights=strict_load_weights)
            else:
                raise ValueError(f"Unknown weights: '{weights}'")

    @property
    def tracing(self):
        return self._tracing

    @tracing.setter
    def tracing(self, is_tracing):
        self._tracing = is_tracing
        self.detr.tracing = is_tracing

    @assert_and_export_onnx(check_mean_std=True, input_mean_std=INPUT_MEAN_STD)
    def forward(self, frames: Union[aloscene.Frame, dict], get_filter_fn: Callable = None, **kwargs):
        """PanopticHead forward, that joint to the previous boxes predictions the new masks feature.

        Parameters
        ----------
        frames : Union[:mod:`Frames <aloscene.frame>`, dict]
            Input frame or DETR/Deformable DETR outputs, with the following parameters :

            - :attr:`pred_logits` : The classification logits (including no-object) for all queries.
            - :attr:`pred_boxes` : The normalized boxes coordinates for all queries, represented as
              (center_x, center_y, height, width). These values are normalized in [0, 1], relative to the size of
              each individual image (disregarding possible padding).
            - :attr:`enc_outputs` : Transformer encoder outputs.
            - :attr:`dec_outputs` : Transformer decoder outputs.
            - :attr:`bb_outputs` : Backbone outputs, in `attr`:`bb_lvl{i}_src_outputs`,`attr`:`bb_lvl{i}_mask_outputs`
              and `attr`:`bb_lvl{i}_pos_outputs` format, with {i} the backbone level.

        get_filter_fn : Callable
            Function that must return two parameters : the :attr:`dec_outputs` tensor filtered by a boolean mask per
            batch (second output). It is expected that the function will at least receive :attr:`frames` and
            :attr:`m_outputs` parameters as input. By default the function used to this purpuse is
            :func:`get_outs_filter` from based model.

        Returns
        -------
        dict
            Output  with the following elements:

            - :attr:`pred_masks` : Binary masks, each one to assign to predicted boxes.
              Shape = [batch_size x num_queries x H // 4 x W // 4]
            - :attr:`pred_masks_info` : Parameters to use in inference procedure
            - :attr:`aux_outputs` : Optional, only returned when auxilary losses are activated. It is a list of
              dictionnaries containing the two above keys for each decoder layer.
            - **:attr:`DETR_outputs`, such as :attr:`pred_logits` and :attr:`pred_boxes`.
        """
        # DETR model forward to obtain box embeddings
        if self.tracing and isinstance(frames, dict):  # Expected export only panoptic Head (without detr)
            assert all([x in frames for x in ["dec_outputs", "enc_outputs", "bb_lvl3_mask_outputs"]])
            assert all([x in frames for x in [f"bb_lvl{i}_src_outputs" for i in range(4)]])
            detr_out = frames  # Expected encode/decode/backbone tensors in frames"
        else:
            detr_out = self.detr_forward(frames, **kwargs)

        proj_src, mask = detr_out["bb_lvl3_src_outputs"], detr_out["bb_lvl3_mask_outputs"]
        bs = proj_src.shape[0]

        if not self.tracing:
            # Filter boxes and get mask indices
            get_filter_fn = get_filter_fn or (
                lambda *args, **kwargs: get_mask_queries(*args, model=self.detr, **kwargs)
            )
            dec_outputs, filters = get_filter_fn(frames=frames, m_outputs=detr_out, **kwargs)
        else:
            # Assume that boxes were filtered previosly / Pass all boxes through network
            dec_outputs, filters = detr_out["dec_outputs"], None
            dec_outputs = dec_outputs[len(dec_outputs) - 1]  # Indexing -1 doesn't work well in torch2onnx

        # Use box embeddings as input of Multi Head attention
        bbox_mask = self.bbox_attention(dec_outputs, detr_out["enc_outputs"], mask=mask)

        # And then, use MHA ouput as input of FPN-style CNN. proj_src = input_proj(features[-1][0])
        seg_masks = self.mask_head(proj_src, bbox_mask, [detr_out[f"bb_lvl{i}_src_outputs"] for i in range(3)[::-1]])
        seg_masks = seg_masks.view(bs, bbox_mask.shape[1], seg_masks.shape[-2], seg_masks.shape[-1])

        # Make output
        forward_head = self.forward_head(seg_masks, detr_outputs=detr_out)

        if self.tracing:  # Return the DETR output + pred_masks if tracing = False
            output = namedtuple("m_outputs", forward_head.keys())
            forward_head = output(*forward_head.values())
        else:
            forward_head["pred_masks_info"] = {"frame_size": frames.shape[-2:], "filters": filters}
        return forward_head

    def forward_head(self, pred_masks: torch.Tensor, detr_outputs: dict, **kwargs):
        """Make the final dictionnary output.

        Parameters
        ----------
        pred_masks : torch.Tensor
            Masks pred by panoptic forward
        detr_outputs : dict
            DETR outputs to append

        Returns
        -------
        dict
            Output describe in :func:`forward` function
        """

        # Minimal outputs
        out = {"pred_masks": pred_masks}
        if self.return_pred_outputs:
            out.update({"pred_logits": detr_outputs["pred_logits"], "pred_boxes": detr_outputs["pred_boxes"]})

        if self.return_detr_outputs:
            out.update({key: val for key, val in detr_outputs.items() if key not in ["pred_logits", "pred_boxes"]})
        return out

    def detr_forward(self, frames: aloscene.Frame, **kwargs):
        """DETR module forward

        Parameters
        ----------
        frames : :mod:`Frames <aloscene.frame>`
            Images batched, of shape [batch_size x 3 x H x W] with a :mod:`Mask <aloscene.mask>`:
            a binary mask of shape [batch_size x 1 x H x W], containing 1 on padded pixels

        Returns
        -------
        dict
            Outputs from the :func:`DETR forward <alonet.detr.detr.Detr.forward>`
        """
        detr_outputs = self.detr(frames, **kwargs)
        if self.tracing and isinstance(detr_outputs, tuple):  # Expected namedtuple
            detr_outputs = detr_outputs._asdict()  # namedtuple to dict
        return detr_outputs

    @torch.no_grad()
    def inference(
        self, forward_out: Dict, maskth: float = 0.5, filters: list = None, frame_size: tuple = None, **kwargs
    ):
        """Given the model forward outputs, this method will return a set of
        :mod:`BoundingBoxes2D <aloscene.bounding_boxes_2d>` and :mod:`Mask <aloscene.mask>`, with its corresponding
        :mod:`Labels <aloscene.labels>` per object detected.

        Parameters
        ----------
        forward_out : dict
            Dict with the model forward outputs
        maskth : float, optional
            Threshold value to binarize the masks, by default 0.5
        filters : list, optional
            List of filter to select the query predicting an object, by default None
        frame_size : tuple, optional
            HW tuple to resize the masks, by default value given in :attr:`forward_out["pred_masks_info"]`

        Returns
        -------
        :mod:`BoundingBoxes2D <aloscene.bounding_boxes_2d>`
            Boxes from DETR model
        :mod:`Mask <aloscene.mask>`
            Binary masks from PanopticHead, one for each box.
        """
        # Get boxes from detr inference and filters
        m_info = forward_out.get("pred_masks_info")
        b_filters = filters or self.detr.get_outs_filter(m_outputs=forward_out, **kwargs)
        m_filters = filters or (m_info.get("filters") if isinstance(m_info, dict) else None)
        if m_filters is None:
            if forward_out["pred_masks"].shape[1] == forward_out["pred_boxes"].shape[1]:
                # Masks not filtered: Index given by bbox position
                b, nq = forward_out["pred_boxes"].shape[:2]
                m_filters = [forward_out["pred_boxes"].new_ones(nq).bool()] * b
            else:
                # Assume that masks were filtered in the following order (bbox_f => boxes filtered):
                # [bbox_f[0] -> mask[0], bbox_f[1] -> mask[1], ..., bbox_f[N] -> mask[N]]
                m_filters = b_filters

        frame_size = frame_size or (m_info.get("frame_size") if isinstance(m_info, dict) else None)
        frame_size = frame_size or forward_out["pred_masks"].shape[-2:]  # Not reshape
        preds_boxes = self.detr.inference(forward_out, filters=b_filters, **kwargs)

        # Procedure to get information about masks
        outputs_masks = forward_out["pred_masks"].squeeze(2)
        if outputs_masks.numel() > 0:
            outputs_masks = F.interpolate(outputs_masks, size=frame_size, mode="bilinear", align_corners=False)
        else:
            outputs_masks = outputs_masks.view(outputs_masks.shape[0], 0, *frame_size)

        # Keep high scores for one-hot encoding
        outputs_masks = F.threshold(outputs_masks.sigmoid(), maskth, 0.0)

        # Transform predictions in aloscene.Mask
        preds_masks = []
        zero_masks = torch.zeros(*frame_size, device=outputs_masks.device, dtype=torch.long)
        for boxes, masks, b_filter, m_filter in zip(preds_boxes, outputs_masks, b_filters, m_filters):

            # One shot encoding, to keep the high score class
            null_values = (~masks.bool()).all(dim=0, keepdim=True)  # Pixels without a score > maskth
            onehot_masks = torch.zeros_like(masks)
            onehot_masks.scatter_(0, masks.argmax(dim=0, keepdim=True), 1)
            masks = onehot_masks.type(torch.long) * (~null_values)  # One-hot encoding with threshold

            # Boxes/masks alignment
            align_masks = []
            m_filter = torch.where(m_filter)[0]
            for ib in torch.where(b_filter)[0]:
                im = (ib == m_filter).nonzero()
                if im.numel() > 0 and im.item() < len(masks):
                    align_masks.append(masks[im.item()])
                else:
                    align_masks.append(zero_masks)

            if len(align_masks) > 0:
                masks = torch.stack(align_masks, dim=0)
            else:
                masks = zero_masks[[]].view(0, *frame_size)

            # Create aloscene object
            masks = aloscene.Mask(masks, names=("N", "H", "W"), labels=boxes.labels)
            preds_masks.append(masks)

        return preds_boxes, preds_masks


def main(image_path):
    from alonet.detr import DetrR50Finetune

    device = torch.device("cuda")

    # Load model
    model = PanopticHead(DetrR50Finetune(num_classes=250), weights="detr-r50-panoptic")
    model.to(device).eval()

    # Open and prepare a batch for the model
    frame = aloscene.Frame(image_path).norm_resnet()
    frames = aloscene.Frame.batch_list([frame])
    frames = frames.to(device)

    # GPU warm up
    [model(frames) for _ in range(3)]

    tic = time.time()
    with torch.no_grad():
        [model(frames) for _ in range(20)]
    toc = time.time()
    print(f"{(toc - tic)/20*1000} ms")

    # Predict boxes/masks
    m_outputs = model(frames)  # Pred of size (B, NQ, H//4, W//4)
    pred_boxes, pred_masks = model.inference(m_outputs, frame_size=frames.HW, threshold=0.85)

    # Add and display the boxes/masks predicted
    frame.append_boxes2d(pred_boxes[0], "pred_boxes")
    frame.append_segmentation(pred_masks[0], "pred_masks")
    frame.get_view().render()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detr R50 Panoptic inference on image")
    parser.add_argument("image_path", type=str, help="Path to the image for inference")
    args = parser.parse_args()
    main(args.image_path)
