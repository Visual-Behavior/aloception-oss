from typing import List
import torch
import torch.nn.functional as F
from torch import nn
import math
import copy

import aloscene
import alonet
from alonet.transformers import MLP

from alonet.detr import Detr
from alonet.deformable_detr.utils import inverse_sigmoid
from alonet.deformable_detr.backbone import Backbone, Joiner
from alonet.deformable_detr.deformable_transformer import (
    DeformableTransformer,
    DeformableTransformerDecoderLayer,
    DeformableTransformerDecoder,
)
from alonet.detr.misc import assert_and_export_onnx


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DeformableDETR(nn.Module):
    """
    The Deformable DETR module for object detection.
    For more details, check its paper https://arxiv.org/abs/2010.04159
    """

    INPUT_MEAN_STD = Detr.INPUT_MEAN_STD

    def __init__(
        self,
        backbone,
        transformer,
        num_classes,
        num_queries=300,
        num_feature_levels=4,
        aux_loss=True,
        with_box_refine=False,
        return_dec_outputs=False,
        return_enc_outputs=False,
        return_bb_outputs=False,
        weights: str = None,
        device: torch.device = torch.device("cuda"),
        activation_fn: str = "sigmoid",
        return_intermediate_dec: bool = True,
        strict_load_weights: bool = True,
    ):
        """Initializes the model.

        Parameters
        ----------
        backbone : nn.Module
            torch module of the backbone to be used. See backbone.py
        transformer : nn.Module
            torch module of the transformer architecture. See transformer.py
        num_classes : int
            number of object classes
        num_queries : int, optional
            number of object queries, ie detection slot. This is the maximal number of objects
            the model can detect in a single image. By default 300
        num_feature_levels : int, optional
            Number of feature map levels will be sampled by multiscale deformable attention. By default 4
        aux_loss : bool, optional
            If True, the model will returns auxilary outputs at each decoder layer
            to calculate auxiliary decoding losses. By default True.
        with_box_refine : bool, optional
            Use iterative box refinement, see paper for more details. By default False
        weights : str, optional
            Pretrained weights, by default None
        device : torch.device, optional
            By default torch.device("cuda")
        activation_fn : str, optional
            Activation function for classification head. Either "sigmoid" or "softmax". By default "sigmoid".

        Notes
        -----
        `activation_fn` = "softmax" implies to work with backgraund class. That means increases in one the num_classes

        """
        super().__init__()
        self.device = device
        self.num_feature_levels = num_feature_levels
        self.transformer = transformer
        self.num_queries = num_queries
        self.return_intermediate_dec = return_intermediate_dec
        self.hidden_dim = transformer.d_model
        self.return_dec_outputs = return_dec_outputs
        self.return_enc_outputs = return_enc_outputs
        self.return_bb_outputs = return_bb_outputs

        if activation_fn not in ["sigmoid", "softmax"]:
            raise Exception(f"activation_fn = {activation_fn} must be one of this two values: 'sigmoid' or 'softmax'.")

        self.activation_fn = activation_fn
        self.background_class = num_classes if self.activation_fn == "softmax" else None
        num_classes += 1 if self.activation_fn == "softmax" else 0  # Add bg class

        self.class_embed = nn.Linear(self.hidden_dim, num_classes)
        self.bbox_embed = MLP(self.hidden_dim, self.hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, self.hidden_dim * 2)
        # Projection for Multi-scale features
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for i in range(num_backbone_outs):
                in_channels = backbone.num_channels[i]
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1), nn.GroupNorm(32, self.hidden_dim),
                    )
                )
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, self.hidden_dim, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, self.hidden_dim),
                    )
                )
                in_channels = self.hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(backbone.num_channels[0], self.hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, self.hidden_dim),
                    )
                ]
            )

        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        num_pred = transformer.decoder.num_layers
        self.num_decoder_layers = transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None

        self.device = device
        if device is not None:
            self.to(device)

        if weights is not None and (
            weights in ["deformable-detr-r50", "deformable-detr-r50-refinement"]
            or ".pth" in weights
            or ".ckpt" in weights
        ):
            alonet.common.load_weights(self, weights, device, strict_load_weights=strict_load_weights)
        else:
            raise ValueError(f"Unknown weights: '{weights}'")

    @assert_and_export_onnx(check_mean_std=True, input_mean_std=INPUT_MEAN_STD)
    def forward(self, frames: aloscene.Frame, **kwargs):
        """
        Deformable DETR

        Parameters
        ----------
        frames: aloscene.Frame
            batched images, of shape [batch_size x 3 x H x W]
            with frames.mask: a binary mask of shape [batch_size x 1 x H x W], containing 1 on padded pixels

        Returns
        -------
        dict
            - "pred_logits": logits classification (including no-object) for all queries.
                            If `self.activation_fn` = "softmax", shape = [batch_size x num_queries x (num_classes + 1)]
                            If `self.activation_fn` = "sigmoid", shape = [batch_size x num_queries x num_classes]
            - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                            (center_x, center_y, height, width). These values are normalized in [0, 1],
                            relative to the size of each individual image (disregarding possible padding).
                            See PostProcess for information on how to retrieve the unnormalized bounding box.
            - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
            - "activation_fn": str, "sigmoid" or "softmax" based on model configuration
        """

        # ==== Backbone
        features, pos = self.backbone(frames, **kwargs)

        assert next(self.parameters()).is_cuda, "DeformableDETR cannot run on CPU (due to MSdeformable op)"

        if "is_export_onnx" in kwargs:
            frame_masks = frames[:, 3:4]
        else:
            frame_masks = frames.mask.as_tensor()

        # ==== Transformer
        srcs = []
        masks = []
        # Project feature maps from the backbone
        for l, feat in enumerate(features):
            src = feat[0]
            mask = (
                feat[1].float()[:, 0].bool()
            )  # Squeeze from (B, 1, H, W) to (B, H, W), casting for TensorRT compability
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        # If the number of ft maps from backbone is less than the required self.num_feature_levels,
        # we project the last feature map
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1][0])
                else:
                    src = self.input_proj[l](srcs[-1])
                mask = F.interpolate(frame_masks.float(), size=src.shape[-2:]).to(torch.bool)
                pos_l = self.backbone[1]((src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask.float()[:, 0].bool())  # [:,0] to squeeze the channel dimension
                pos.append(pos_l)

        query_embeds = self.query_embed.weight
        transformer_outptus = self.transformer(srcs, masks, pos, query_embeds, **kwargs)
        return self.forward_heads(transformer_outptus, bb_outputs=features)

    def forward_position_heads(self, transformer_outptus):
        hs = transformer_outptus["hs"]
        init_reference = transformer_outptus["init_reference_out"]
        inter_references = transformer_outptus["inter_references_out"]

        outputs_coords = []

        for lvl in range(hs.shape[0]):

            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]

            reference = inverse_sigmoid(reference)
            tmp = self.bbox_embed[lvl](hs[lvl])

            if reference.shape[-1] == 4:  # Refinment
                tmp += reference
            else:
                assert reference.shape[-1] == 2  # None refinment
                tmp[..., :2] += reference

            outputs_coord = tmp.sigmoid()  # get normalized value ranging from 0 to 1
            outputs_coords.append(outputs_coord)

        outputs_coord = torch.stack(outputs_coords)

        return outputs_coords

    def forward_class_heads(self, transformer_outptus):
        hs = transformer_outptus["hs"]
        init_reference = transformer_outptus["init_reference_out"]
        inter_references = transformer_outptus["inter_references_out"]

        # ==== Detection head
        outputs_classes = []
        for lvl in range(hs.shape[0]):
            outputs_class = self.class_embed[lvl](hs[lvl])
            outputs_classes.append(outputs_class)

        outputs_class = torch.stack(outputs_classes)
        return outputs_class

    def forward_heads(self, transformer_outptus: dict, bb_outputs: list = None, **kwargs):
        """Apply Deformable heads"""
        outputs_class = self.forward_class_heads(transformer_outptus)
        outputs_coord = self.forward_position_heads(transformer_outptus)

        out = {
            "pred_logits": outputs_class[self.num_decoder_layers - 1],
            "pred_boxes": outputs_coord[self.num_decoder_layers - 1],
            "activation_fn": self.activation_fn,
        }

        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord, activation_fn=out["activation_fn"])

        if self.return_dec_outputs:
            out["dec_outputs"] = transformer_outptus["hs"]

        if self.return_enc_outputs:
            out["enc_outputs"] = transformer_outptus["memory"]

        if self.return_bb_outputs:
            out["bb_outputs"] = bb_outputs

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, **kwargs):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"pred_logits": a, "pred_boxes": b, **kwargs} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def get_outs_labels(self, m_outputs: dict = None, activation_fn: str = None,) -> List[torch.Tensor]:
        """Given the model outs_scores and the model outs_labels,
        return the labels and the associated scores.

        Parameters
        ----------
        m_outputs : dict, optional
            Dict of forward outputs, by default None
        threshold : float, optional
            Score threshold if sigmoid activation is used. By default 0.2
        activation_fn : str, optional
            Either "sigmoid" or "softmax". By default None.
            If "sigmoid" is used, filter is based on score threshold.
            If "softmax" is used, filter is based on non-background classes.

        Returns
        -------
        Tuple
            (torch.Tensor, torch.Tensor) being the predicted labels and scores
        """
        activation_fn = activation_fn or self.activation_fn
        assert (m_outputs) is not None
        if "activation_fn" not in m_outputs:
            raise Exception("'activation_fn' must be declared in forward output.")
        activation_fn = m_outputs["activation_fn"]
        if activation_fn == "softmax":
            outs_probs = F.softmax(m_outputs["pred_logits"], -1)
        else:
            outs_probs = m_outputs["pred_logits"].sigmoid()
        outs_scores, outs_labels = outs_probs.max(-1)
        return outs_labels, outs_scores

    def get_outs_filter(
        self,
        outs_scores: torch.Tensor = None,
        outs_labels: torch.Tensor = None,
        m_outputs: dict = None,
        threshold=None,
        activation_fn: str = None,
    ) -> List[torch.Tensor]:
        """Given the model outs_scores and the model outs_labels,
        return a list of filter for each output. If `out_scores` and `outs_labels` are not provided,
        the method will rely on the model forward outputs `m_outputs` to extract the `outs_scores`
        and the `outs_labels` on its own.


        Parameters
        ----------
        outs_scores : torch.Tensor, optional
            Predicted scores, by default None
        outs_labels : torch.Tensor, optional
            Predicted labels, by default None
        m_outputs : dict, optional
            Dict of forward outputs, by default None
        threshold : float, optional
            Score threshold to use. if None and sigmoid is used, 0.2 will be used as default value.
        softmax_threshold: float, optinal
            Score threshold if softmax activation is used. None by default.
        activation_fn : str, optional
            Either "sigmoid" or "softmax". By default None.
            If "sigmoid" is used, filter is based on score threshold.
            If "softmax" is used, filter is based on non-background classes.

        Returns
        -------
        List[torch.Tensor]
            List of filter to select the query predicting an object, len = batch size
        """
        activation_fn = activation_fn or self.activation_fn

        if outs_scores is None or outs_labels is None:
            outs_labels, outs_scores = self.get_outs_labels(m_outputs, activation_fn=activation_fn)

        filters = []
        for scores, labels in zip(outs_scores, outs_labels):
            if activation_fn == "softmax":
                softmax_threshold = threshold
                if softmax_threshold is None:
                    filters.append(labels != self.background_class)
                else:
                    filters.append((labels != self.background_class) & (scores > softmax_threshold))
            else:
                sigmoid_threshold = 0.2 if threshold is None else threshold
                filters.append(scores > sigmoid_threshold)
        return filters

    @torch.no_grad()
    def inference(self, forward_out: dict, threshold=0.2, filters=None, **kwargs):
        """Get model outptus as returned by the the forward method"""
        outs_logits, outs_boxes = forward_out["pred_logits"], forward_out["pred_boxes"]

        if "activation_fn" not in forward_out:
            raise Exception("'activation_fn' must be declared in forward output.")
        activation_fn = forward_out["activation_fn"]

        if activation_fn == "softmax":
            outs_probs = F.softmax(outs_logits, -1)
        else:
            outs_probs = outs_logits.sigmoid()
        outs_scores, outs_labels = outs_probs.max(-1)

        if filters is None:
            filters = self.get_outs_filter(
                outs_scores=outs_scores,
                outs_labels=outs_labels,
                threshold=threshold,
                activation_fn=activation_fn,
                **kwargs,
            )

        preds_boxes = []
        for scores, labels, boxes, b_filter in zip(outs_scores, outs_labels, outs_boxes, filters):
            boxes = boxes[b_filter]
            labels = labels[b_filter]
            scores = scores[b_filter]
            boxes_labels = aloscene.Labels(labels.type(torch.float32), encoding="id", scores=scores, names=("N",))
            boxes = aloscene.BoundingBoxes2D(
                boxes.cpu(), boxes_format="xcyc", absolute=False, names=("N", None), labels=boxes_labels
            )
            preds_boxes.append(boxes)

        return preds_boxes

    def build_positional_encoding(self, hidden_dim=256):
        N_steps = hidden_dim // 2
        position_embed = alonet.transformers.PositionEmbeddingSine(N_steps, normalize=True, center=True)
        return position_embed

    def build_backbone(
        self,
        backbone_name: str = "resnet50",
        train_backbone: bool = True,
        return_interm_layers: bool = True,
        dilation: bool = False,
    ):
        """
        Build backbone for Deformable DETR

        Parameters
        ----------
        backbone_name : str, optional
            name in torchvision.models, by default "resnet50"
        train_backbone : bool, optional
            By default True
        return_interm_layers : bool, optional
            Needed if we use segmentation or multi-scale, by default True
        dilation : bool, optional
            If True, we replace stride with dilation in the last convolutional block (DC5). By default False.

        Returns
        -------
        alonet.deformable_detr.backbone.Backbone
            Resnet backbone
        """

        return Backbone(backbone_name, train_backbone, return_interm_layers, dilation)

    def build_decoder_layer(
        self,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        nheads: int = 8,
        dim_feedforward: int = 1024,
        num_feature_levels: int = 4,
        dec_n_points: int = 4,
    ):

        return DeformableTransformerDecoderLayer(
            d_model=hidden_dim,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            n_levels=num_feature_levels,
            n_heads=nheads,
            n_points=dec_n_points,
        )

    def build_decoder(
        self, dec_layers: int = 6, return_intermediate_dec=True,
    ):

        decoder_layer = self.build_decoder_layer()

        return DeformableTransformerDecoder(decoder_layer, dec_layers, return_intermediate_dec)

    def build_transformer(
        self,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        nheads: int = 8,
        dim_feedforward: int = 1024,
        enc_layers: int = 6,
        dec_layers: int = 6,
        num_feature_levels: int = 4,
        dec_n_points: int = 4,
        enc_n_points: int = 4,
        return_intermediate_dec=True,
    ):

        decoder = self.build_decoder()

        return DeformableTransformer(
            decoder=decoder,
            d_model=hidden_dim,
            nhead=nheads,
            dropout=dropout,
            num_decoder_layers=dec_layers,
            num_encoder_layers=enc_layers,
            dim_feedforward=dim_feedforward,
            num_feature_levels=num_feature_levels,
            dec_n_points=dec_n_points,
            enc_n_points=enc_n_points,
            return_intermediate_dec=return_intermediate_dec,
        )


def build_deformable_detr_r50(
    num_classes=91, box_refinement=True, weights=None, device=torch.device("cuda")
) -> DeformableDETR:
    """[summary]

    Parameters
    ----------
    num_classes : int, optional
        Number of classes for objection detection, by default 91
    box_refinement : bool, optional
        Use iterative box refinement, by default True
    weights : str, optional
        Pretrained weights, by default None
    device : torch.device, optional
        By default torch.device("cuda")

    Returns
    -------
    DeformableDETR
    """
    backbone = DeformableDETR.build_backbone()
    position_embed = DeformableDETR.build_positional_encoding()
    backbone = Joiner(backbone, position_embed)
    transformer = DeformableDETR.build_transformer()
    return DeformableDETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_feature_levels=4,
        aux_loss=False,
        with_box_refine=box_refinement,
        weights=weights,
        device=device,
    )
