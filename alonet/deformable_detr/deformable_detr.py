"""
The Deformable DETR module for object detection.
For more details, check its paper https://arxiv.org/abs/2010.04159
"""
from typing import List
import torch
import torch.nn.functional as F
from torch import nn
import math
import copy
from collections import namedtuple

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
        Activation function for classification head. Either ``sigmoid`` or ``softmax``. By default "sigmoid".
    tracing : bool, Optional
        Change model behavior to be exported as TorchScript, by default False

    Notes
    -----
    ``activation_fn`` = ``softmax`` implies to work with backgraund class. That means increases in one the num_classes

    """

    INPUT_MEAN_STD = Detr.INPUT_MEAN_STD

    def __init__(
        self,
        backbone: nn,
        transformer: nn,
        num_classes: int,
        num_queries: int = 300,
        num_feature_levels: int = 4,
        aux_loss: bool = True,
        with_box_refine: bool = False,
        return_dec_outputs: bool = False,
        return_enc_outputs: bool = False,
        return_bb_outputs: bool = False,
        weights: str = None,
        device: torch.device = torch.device("cuda"),
        activation_fn: str = "sigmoid",
        return_intermediate_dec: bool = True,
        strict_load_weights: bool = True,
        tracing=False,
        add_depth=False,
    ):
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
        self.add_depth = add_depth

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
            num_backbone_outs = len(backbone.strides) - 1  # Ignore backbone.layer1
            input_proj_list = []
            for i in range(1, num_backbone_outs + 1):  # Ignore backbone.layer1
                in_channels = backbone.num_channels[i] + (1 if add_depth else 0)
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, self.hidden_dim),
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
                        nn.Conv2d(backbone.num_channels[0] + (1 if add_depth else 0), self.hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, self.hidden_dim),
                    )
                ]
            )

        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.tracing = tracing

        if self.tracing and self.aux_loss:
            raise AttributeError("When tracing = True, aux_loss must be False")

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

        if weights is not None:
            if (
                weights in ["deformable-detr-r50", "deformable-detr-r50-refinement"]
                or ".pth" in weights
                or ".ckpt" in weights
            ):
                alonet.common.load_weights(self, weights, device, strict_load_weights=strict_load_weights)
            else:
                raise ValueError(f"Unknown weights: '{weights}'")

    @property
    def tracing(self):
        return self._tracing

    @tracing.setter
    def tracing(self, is_tracing):
        self._tracing = is_tracing
        self.backbone.tracing = is_tracing

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
            It outputs a dict with the following elements:

            - :attr:`pred_logits`: The classification logits (including no-object) for all queries.
              Shape= [batch_size x num_queries x (num_classes + 1)]
            - :attr:`pred_boxes`: The normalized boxes coordinates for all queries, represented as
              (center_x, center_y, height, width). These values are normalized in [0, 1], relative to the size of
              each individual image (disregarding possible padding).
              See PostProcess for information on how to retrieve the unnormalized bounding box.
            - :attr:`activation_fn`: str, ``sigmoid`` or ``softmax`` based on model configuration
            - :attr:`aux_outputs`: Optional, only returned when auxilary losses are activated. It is a list of
              dictionnaries containing the two above keys for each decoder layer.
            - :attr:`bb_outputs`: Optional, only returned when backbone outputs are activated.
            - :attr:`enc_outputs`: Optional, only returned when transformer encoder outputs are activated.
            - :attr:`dec_outputs`: Optional, only returned when transformer decoder outputs are activated.
        """

        # ==== Backbone
        features, pos = self.backbone(frames, **kwargs)

        assert next(self.parameters()).is_cuda, "DeformableDETR cannot run on CPU (due to MSdeformable op)"

        if self.tracing:
            frame_masks = frames[:, 3:4]
        else:
            frame_masks = frames.mask.as_tensor()

        # ==== Transformer
        srcs = []
        masks = []
        depth = []
        # Project feature maps from the backbone
        for lf, feat in enumerate(features[1:]):  # bacbone.layer1 ignored
            src = feat[0]
            mask = (
                feat[1].float()[:, 0].bool()
            )  # Squeeze from (B, 1, H, W) to (B, H, W), casting for TensorRT compability
            srcs.append(self.input_proj[lf](src))
            masks.append(mask)
            depth.append(src[:, -1:])
            assert mask is not None
        # If the number of ft maps from backbone is less than the required self.num_feature_levels,
        # we project the last feature map
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for lf in range(_len_srcs, self.num_feature_levels):
                if lf == _len_srcs:
                    depth.append(torch.nn.functional.max_pool2d(features[-1][0][:, -1:], 3, 2, 1))
                    src = self.input_proj[lf](features[-1][0])
                else:
                    depth.append(torch.nn.functional.max_pool2d(srcs[-1, -1:], 3, 2, 1))
                    src = self.input_proj[lf](srcs[-1])
                mask = F.interpolate(frame_masks.float(), size=src.shape[-2:]).to(torch.bool)
                pos_l = self.backbone[1]((src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask.float()[:, 0].bool())  # [:,0] to squeeze the channel dimension
                pos.append(pos_l)

        query_embeds = self.query_embed.weight
        transformer_outptus = self.transformer(
            srcs, masks, pos[1:], query_embeds, depth if self.add_depth else None, **kwargs
        )

        # Feature reconstruction with features[-1][0] = input_proj(features[-1][0])
        if self.return_bb_outputs:
            features[-1] = (srcs[-2], masks[-2])
        forward_head = self.forward_heads(transformer_outptus, bb_outputs=(features, pos[:-1]))

        if self.tracing:
            forward_head.pop("activation_fn")  # Not include in exportation
            output = namedtuple("m_outputs", forward_head.keys())
            forward_head = output(*forward_head.values())
        return forward_head

    def forward_position_heads(self, transformer_outptus: dict):
        """Forward from transformer decoder output into positional (boxes)

        Parameters
        ----------
        transformer_outptus : dict
            Output of transformer layer

        Returns
        -------
        torch.Tensor
            Output of shape [batch_size x num_queries x 4]
        """

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

    def forward_class_heads(self, transformer_outptus: dict):
        """Forward from transformer decoder output into class_embed layer to get class predictions

        Parameters
        ----------
        transformer_outptus : dict
            Output of transformer layer

        Returns
        -------
        torch.Tensor
            Output of shape [batch_size x num_queries x (num_classes + 1)]
        """
        hs = transformer_outptus["hs"]
        # init_reference = transformer_outptus["init_reference_out"]
        # inter_references = transformer_outptus["inter_references_out"]

        # ==== Detection head
        outputs_classes = []
        for lvl in range(hs.shape[0]):
            outputs_class = self.class_embed[lvl](hs[lvl])
            outputs_classes.append(outputs_class)

        outputs_class = torch.stack(outputs_classes)
        return outputs_class

    def forward_heads(self, transformer_outptus: dict, bb_outputs: list = None, **kwargs):
        """Apply Deformable heads and make the final dictionnary output.

        Parameters
        ----------
        transformer_outptus : dict
            Output of transformer layer
        bb_outputs : torch.Tensor, optional
            Backbone output to append in output, by default None

        Returns
        -------
        dict
            Output describe in :func:`forward` function
        """
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
            out["enc_outputs"] = transformer_outptus["memory"][-2]  # Encoder layer used from PanopticHead

        if self.return_bb_outputs:
            features, pos = bb_outputs
            for lvl, (src, mask) in enumerate(features):
                out[f"bb_lvl{lvl}_src_outputs"] = src
                out[f"bb_lvl{lvl}_mask_outputs"] = mask
                out[f"bb_lvl{lvl}_pos_outputs"] = pos[lvl]

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, **kwargs):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"pred_logits": a, "pred_boxes": b, **kwargs} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def get_outs_labels(
        self,
        m_outputs: dict = None,
        activation_fn: str = None,
    ) -> List[torch.Tensor]:
        """Given the model outs_scores and the model outs_labels,
        return the labels and the associated scores.

        Parameters
        ----------
        m_outputs : dict, optional
            Dict of forward outputs, by default None
        threshold : float, optional
            Score threshold if sigmoid activation is used. By default 0.2
        activation_fn : str, optional
            Either ``sigmoid`` or ``softmax``. By default None.

            - If ``sigmoid`` is used, filter is based on score threshold.
            - If ``softmax`` is used, filter is based on non-background classes.

        Returns
        -------
        Tuple
            (torch.Tensor, torch.Tensor) being the predicted labels and scores
        """
        assert (m_outputs) is not None
        activation_fn = m_outputs.get("activation_fn") or activation_fn or self.activation_fn
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
        """Given the model ``outs_scores`` and the model ``outs_labels``,
        return a list of filter for each output. If ``out_scores`` and ``outs_labels`` are not provided,
        the method will rely on the model forward outputs ``m_outputs`` to extract the ``outs_scores``
        and the ``outs_labels`` on its own.


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
            Either ``sigmoid`` or ``softmax``. By default None.

            - If ``sigmoid`` is used, filter is based on score threshold.
            - If ``softmax`` is used, filter is based on non-background classes.

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
        """Given the model forward outputs, this method will return an
        :mod:`BoundingBoxes2D <aloscene.bounding_boxes_2d>` tensor.

        Parameters
        ----------
        forward_out : dict
            Dict with the model forward outptus
        filters : list
            list of torch.Tensor will a filter on which prediction to select to create the set
            of :mod:`BoundingBoxes2D <aloscene.bounding_boxes_2d>`.

        Returns
        -------
        boxes : :mod:`BoundingBoxes2D <aloscene.bounding_boxes_2d>`
            Boxes filtered and predicted by forward outputs
        """
        outs_logits, outs_boxes = forward_out["pred_logits"], forward_out["pred_boxes"]
        activation_fn = forward_out.get("activation_fn") or self.activation_fn

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

    def build_positional_encoding(self, hidden_dim: int = 256):
        """Build the positinal encoding layer to combine input values with respect to theirs position

        Parameters
        ----------
        hidden_dim : int, optional
            Hidden dimension size, by default 256

        Returns
        -------
        torch.nn
            Default architecture to encode input with values and theirs position
        """
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
        add_cross_attn_channel=False,
    ):
        """Build decoder layer

        Parameters
        ----------
        hidden_dim : int, optional
            Hidden dimension size, by default 256
        dropout : float, optional
            Dropout value, by default 0.1
        nheads : int, optional
            Number of heads, by default 8
        dim_feedforward : int, optional
            Feedfoward dimension size, by default 2048
        normalize_before : bool, optional
            Use normalize before each layer, by default False
        dec_n_points : int, optional
            Number of points use in deformable layer, by default 4

        Returns
        -------
        :class:`~alonet.deformable.deformable_transformer.DeformableTransformerDecoderLayer`
            Transformer decoder layer
        """
        return DeformableTransformerDecoderLayer(
            d_model=hidden_dim,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            n_levels=num_feature_levels,
            n_heads=nheads,
            n_points=dec_n_points,
            add_cross_attn_channel=add_cross_attn_channel,
        )

    def build_decoder(self, dec_layers: int = 6, return_intermediate_dec: bool = True, add_cross_attn_channel=False):
        """Build decoder layer

        Parameters
        ----------
        dec_layers : int, optional
            Number of decoder layers, by default 6
        return_intermediate_dec : bool, optional
            Return intermediate decoder outputs, by default True

        Returns
        -------
        :class:`~alonet.deformable.deformable_transformer.DeformableTransformerDecoder`
            Transformer decoder
        """
        decoder_layer = self.build_decoder_layer(add_cross_attn_channel=add_cross_attn_channel)

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
        return_intermediate_dec: bool = True,
        add_depth=False,
    ):
        """Build transformer

        Parameters
        ----------
        hidden_dim : int, optional
            Hidden dimension size, by default 256
        dropout : float, optional
            Dropout value, by default 0.1
        nheads : int, optional
            Number of heads, by default 8
        dim_feedforward : int, optional
            Feedfoward dimension size, by default 2048
        enc_layers : int, optional
            Number of encoder layers, by default 6
        dec_layers : int, optional
            Number of decoder layers, by default 6
        num_feature_levels : int, optional
            Number of feature map levels will be sampled by multiscale deformable attention. By default 4
        dec_n_points : int, optional
            Number of points use in deformable decoder layer, by default 4
        enc_n_points : int, optional
            Number of points use in deformable encoder layer, by default 4
        return_intermediate_dec : bool, optional
            Return intermediate decoder outputs, by default True

        Returns
        -------
        :mod:`Transformer <alonet.detr.transformer>`
            Transformer module
        """
        decoder = self.build_decoder(add_cross_attn_channel=add_depth)

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
            dec_depth=add_depth,
        )


def build_deformable_detr_r50(
    num_classes: int = 91, box_refinement: bool = True, weights: bool = None, device=torch.device("cuda")
) -> DeformableDETR:
    """Build a deformable DETR architecture, based on RESNET50

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
