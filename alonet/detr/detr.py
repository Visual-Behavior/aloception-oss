# Inspired by the official DETR repository and adapted for aloception
"""
End-to-End Object Detection with Transformers (DETR) model.
"""
import torch
import torch.nn.functional as F
from torch import nn

from alonet.detr.transformer import Transformer
from alonet.transformers import MLP, PositionEmbeddingSine
from alonet.detr.backbone import Backbone
import alonet
import aloscene
from alonet.detr.misc import assert_and_export_onnx

INPUT_MEAN_STD = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


class Detr(nn.Module):
    """This is the DETR module that performs object detection

    Parameters
    ----------
    backbone : torch.module
        Torch module of the backbone to be used. See backbone.py
    transformer : torch.module
        Torch module of the transformer architecture. See transformer.py
    num_classes : int
        number of object classes
    num_queries : int
        number of object queries, ie detection slot. This is the maximal number of objects
        DETR can detect in a single image. For COCO, we recommend 100 queries.
    background_class : int, Optional
        If none, the background_class will automaticly be set to be equal to the num_classes.
        In other word, by default, the background class will be set as the last class of the model
    weights : str, Optional
        Load weights from path or support :attr:`model_name`, by default None
    device : torch.device, Optional
        Architecture makes in a specific device, by default torch.device("cpu")
    aux_loss : bool, Optional
        True if auxiliary decoding losses (loss at each decoder layer) are to be used, by default True
    return_dec_outputs : bool, Optional
        If True, the dict output will contains a key : "dec_outputs"
        with the decoder outputs of shape (stage, batch, num_queries, dim), by default False
    return_enc_outputs : bool, Optional
        If True, the dict output will contains a key : "enc_outputs"
        with the encoder outputs of shape (num_enc, stage, HB, WB), by default False
    return_bb_outputs : bool, Optional
        If True, the dict output will contains a key : "bb_outputs"
        with the the list of the different backbone outputs, by default False
    strict_load_weights : bool, Optional
        Load the weights (if any given) with strict=True, by default True
    tracing : bool, Optional
        Change model behavior to be exported as TorchScript, by default False
    """

    INPUT_MEAN_STD = INPUT_MEAN_STD

    def __init__(
        self,
        backbone,
        transformer,
        num_classes,
        num_queries,
        background_class: int = None,
        aux_loss=True,
        weights: str = None,
        return_dec_outputs=False,
        return_enc_outputs=False,
        return_bb_outputs=False,
        device: torch.device = torch.device("cpu"),
        strict_load_weights=True,
        tracing: bool = False,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.num_decoder_layers = transformer.decoder.num_layers
        self.num_classes = num_classes
        self.return_dec_outputs = return_dec_outputs
        self.tracing = tracing
        self.return_enc_outputs = return_enc_outputs
        self.return_bb_outputs = return_bb_outputs

        # +1 on the num of class because Detr use softmax, and the background class
        # is by default assume to be the last element. (Except if background_class is set to be different.
        self.background_class = self.num_classes if background_class is None else background_class
        self.num_classes += 1

        hidden_dim = transformer.d_model
        self.hidden_dim = hidden_dim

        self.class_embed = self.build_class_embed()
        self.bbox_embed = self.build_bbox_embed()

        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.backbone.tracing = tracing
        self.aux_loss = aux_loss

        if device is not None:
            self.to(device)

        if weights is not None:
            if weights == "detr-r50" or ".pth" in weights or ".ckpt" in weights:
                alonet.common.load_weights(self, weights, device, strict_load_weights=strict_load_weights)
            else:
                raise ValueError(f"Unknown weights: '{weights}'")

        self.device = device
        self.INPUT_MEAN_STD = INPUT_MEAN_STD

    @assert_and_export_onnx(check_mean_std=True, input_mean_std=INPUT_MEAN_STD)
    def forward(self, frames: aloscene.Frame, **kwargs):
        """Detr model forward

        Parameters
        ----------
        frames : :mod:`Frames <aloscene.frame>`
            Images batched, of shape [batch_size x 3 x H x W] with a :mod:`Mask <aloscene.mask>`:
            a binary mask of shape [batch_size x 1 x H x W], containing 1 on padded pixels

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
            - :attr:`aux_outputs`: Optional, only returned when auxilary losses are activated. It is a list of
              dictionnaries containing the two above keys for each decoder layer.
            - :attr:`bb_outputs`: Optional, only returned when backbone outputs are activated.
            - :attr:`enc_outputs`: Optional, only returned when transformer encoder outputs are activated.
            - :attr:`dec_outputs`: Optional, only returned when transformer decoder outputs are activated.
        """
        features, pos = self.backbone(frames, **kwargs)
        src, mask = features[-1][0], features[-1][1]
        # assert len(mask.shape) == 4
        # assert mask.shape[1] == 1
        # mask = torch.squeeze(mask, dim=1)
        # Pytorch will complicate the Squeeze op when exporting to ONNX
        # So  we should use slicing instead of Squeeze
        mask = mask.to(torch.float32)  # TensorRT doesn't support slicing/gathering on bool
        mask = mask[:, 0]
        mask = mask.to(torch.bool)

        input_proj = self.input_proj(src)
        transformer_outptus = self.transformer(input_proj, mask, self.query_embed.weight, pos[-1], **kwargs)

        # Feature reconstruction with features[-1][0] = input_proj(features[-1][0])
        if self.return_bb_outputs:
            features[-1] = (input_proj, mask)

        forward_head = self.forward_heads(transformer_outptus, bb_outputs=(features, pos))
        if self.tracing:
            forward_head = (forward_head["pred_boxes"], forward_head["pred_logits"])
        return forward_head

    def forward_position_heads(self, transformer_outptus: dict):
        """Forward from transformer decoder output into bbox_embed layer to get box predictions

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
        return self.bbox_embed(hs).sigmoid()

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
        outputs_class = self.class_embed(hs)
        return outputs_class

    def forward_heads(self, transformer_outptus: dict, bb_outputs: torch.Tensor = None, **kwargs):
        """Apply Detr heads and make the final dictionnary output.

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
            # Indexing -1 doesn't work well in torch2onnx
            "pred_logits": outputs_class[self.num_decoder_layers - 1],
            "pred_boxes": outputs_coord[self.num_decoder_layers - 1],
        }
        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)

        if self.return_dec_outputs:
            out["dec_outputs"] = transformer_outptus["hs"]

        if self.return_enc_outputs:
            out["enc_outputs"] = transformer_outptus["memory"]

        if self.return_bb_outputs:
            out["bb_outputs"] = bb_outputs

        return out

    def get_outs_labels(self, m_outputs: dict):
        """This method will return the label and class of each slot.

        Parameters
        ----------
        m_outputs: dict
            Model forward output

        Returns
        -------
        labels: torch.Tensor
            predictec class for each slot
        scores: torch.Tensor
            predicted score for each slot
        """
        outs_logits = m_outputs["pred_logits"]
        outs_probs = F.softmax(outs_logits, -1)
        outs_scores, outs_labels = outs_probs.max(-1)
        return outs_labels, outs_scores

    def get_outs_filter(
        self,
        outs_scores: torch.Tensor = None,
        outs_labels: torch.Tensor = None,
        m_outputs: dict = None,
        background_class: int = None,
        threshold: float = None,
        **kwargs,
    ):
        """Given the model outs_scores and the model outs_labels whit method return a list of filter for each output.
        If :attr:`out_scores` and :attr:`outs_labels` are not provided, the method will rely on the model forward
        outputs (:attr:`m_outputs`) to extract the :attr:`outs_scores` and the :attr:`outs_labels` on its own.

        Parameters
        ----------
        outs_scores : torch.Tensor, Optional
            Output score from :func:`forward`, by default None
        outs_labels : torch.Tensor, Optional
            Output labels from :func:`forward`, by default None
        m_outputs : dict, Optional
            Forward outputs, by default None
        background_class : int, Optional
            ID background class, used to filter classes, by default :attr:`background_class` defined in constructor
        threshold : float, Optional
            Threshold value to filter classes by score, by default not implement

        Returns
        -------
        filters: list
            List of filter to select the query predicting an object.
        """
        background_class = background_class or self.background_class
        if outs_scores is None or outs_labels is None:
            assert (m_outputs) is not None
            outs_labels, outs_scores = self.get_outs_labels(m_outputs)

        filters = []
        for scores, labels in zip(outs_scores, outs_labels):
            if threshold is None:
                filters.append(labels != background_class)
            else:
                filters.append((labels != background_class) & (scores > threshold))

        return filters

    @torch.no_grad()
    def inference(self, forward_out: dict, filters=None, background_class=None, threshold=None):
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
        outs_probs = F.softmax(outs_logits, -1)
        outs_scores, outs_labels = outs_probs.max(-1)

        if filters is None:
            filters = self.get_outs_filter(
                outs_scores=outs_scores,
                outs_labels=outs_labels,
                background_class=background_class,
                threshold=threshold,
            )

        preds_boxes = []
        for scores, labels, boxes, b_filter in zip(outs_scores, outs_labels, outs_boxes, filters):
            scores = scores[b_filter]
            boxes = boxes[b_filter]
            labels = labels[b_filter]
            boxes_labels = aloscene.Labels(labels.type(torch.float32), encoding="id", scores=scores, names=("N",))
            boxes = aloscene.BoundingBoxes2D(
                boxes, boxes_format="xcyc", absolute=False, names=("N", None), labels=boxes_labels
            )
            preds_boxes.append(boxes)

        return preds_boxes

    def build_class_embed(self):
        """Layer defined to class embed

        Returns
        -------
        torch.nn
            Class embed layer
        """
        return nn.Linear(self.hidden_dim, self.num_classes)

    def build_bbox_embed(self):
        """MLP implemented to predict boxes coordinates

        Returns
        -------
        torch.nn
            Multi-Layer perceptron with 4 neurons in last layer
        """
        return MLP(self.hidden_dim, self.hidden_dim, 4, 3)

    def build_positional_encoding(self, hidden_dim: int = 256, position_embedding: str = "sin", center: bool = False):
        """Build the positinal encoding layer to combine input values with respect to theirs position

        Parameters
        ----------
        hidden_dim : int, optional
            Hidden dimension size, by default 256
        position_embedding : str, optional
            Position encoding type, by default "sin"
        center : bool, optional
            Use center in position encoding, by default False

        Returns
        -------
        torch.nn
            Default architecture to encode input with values and theirs position

        Raises
        ------
        NotImplementedError
            :attr:`v3` and :attr:`learned` encoding types not support yet
        ValueError
            Support only :attr:`v2` and :attr:`sine` encoding types
        """
        # Positional encoding
        position_embedding = "sine"
        hidden_dim = 256
        N_steps = hidden_dim // 2
        if position_embedding in ("v2", "sine"):
            # TODO find a better way of exposing other arguments
            position_embedding = PositionEmbeddingSine(N_steps, normalize=True, center=center)
        elif position_embedding in ("v3", "learned"):
            raise NotImplementedError()
        else:
            raise ValueError(f"not supported {position_embedding}")
        return position_embedding

    def build_backbone(
        self,
        backbone_name: str,
        train_backbone: bool,
        return_interm_layers: bool,
        dilation: bool,
        aug_tensor_compatible: bool = True,
    ):
        """Build backbone architecture

        Parameters
        ----------
        backbone_name : str
            Backbone name
        train_backbone : bool
            Train backbone parameters if required
        return_interm_layers : bool
            Return intermediate layers if required
        dilation : bool
            Use dilation
        aug_tensor_compatible : bool, optional
            Compatibility with augmented tensors, by default True

        Returns
        -------
        :mod:`Backbone <alonet.detr.backbone>`
            Architecture used to encode input images
        """
        return Backbone(
            backbone_name, train_backbone, return_interm_layers, dilation, aug_tensor_compatible=aug_tensor_compatible
        )

    def build_decoder_layer(
        self,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        nheads: int = 8,
        dim_feedforward: int = 2048,
        normalize_before: bool = False,
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
            use normalize before each layer, by default False

        Returns
        -------
        :class:`TransformerDecoderLayer <alonet.detr.transformer.TransformerDecoderLayer>`
            Transformer decoder layer
        """
        return alonet.detr.transformer.TransformerDecoderLayer(
            d_model=hidden_dim,
            n_heads=nheads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            normalize_before=normalize_before,
        )

    def build_decoder(
        self, hidden_dim: int = 256, num_decoder_layers: int = 6,
    ):
        """Build decoder layer

        Parameters
        ----------
        hidden_dim : int, optional
            Hidden dimension size, by default 256
        num_decoder_layers : int, optional
            Number of decoder layers, by default 6

        Returns
        -------
        :class:`TransformerDecoder <alonet.detr.transformer.TransformerDecoder>`
            Transformer decoder
        """
        decoder_layer = self.build_decoder_layer()

        return alonet.detr.transformer.TransformerDecoder(
            decoder_layer, num_decoder_layers, nn.LayerNorm(hidden_dim), return_intermediate=True
        )

    def build_transformer(
        self,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        nheads: int = 8,
        dim_feedforward: int = 2048,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        normalize_before: bool = False,
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
        num_encoder_layers : int, optional
            Number of encoder layers, by default 6
        num_decoder_layers : int, optional
            Number of decoder layers, by default 6
        normalize_before : bool, optional
            use normalize before each layer, by default False

        Returns
        -------
        :mod:`Transformer <alonet.detr.transformer>`
            Transformer module
        """
        decoder = self.build_decoder()

        return Transformer(
            d_model=hidden_dim,
            dropout=dropout,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            normalize_before=normalize_before,
            return_intermediate_dec=True,
            decoder=decoder,
        )

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"pred_logits": a, "pred_boxes": b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
