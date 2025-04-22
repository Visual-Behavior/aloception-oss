# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import Tensor
import torchvision.transforms.functional as tvF
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List, Union


from alonet.transformers.position_encoding import build_position_encoding
import aloscene
from alonet.detr.misc import assert_and_export_onnx

# Bellow some usefull distributed method
# to move something else
import torch.distributed as dist


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):
    """Base class to define behavior of backbone
    """

    def __init__(
        self,
        backbone: nn.Module,
        train_backbone: bool,
        num_channels: int,
        return_interm_layers: bool,
        aug_tensor_compatible: bool = True,
        tracing: bool = False,
    ):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or "layer2" not in name and "layer3" not in name and "layer4" not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {"layer4": "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels
        self.tracing = tracing

    @assert_and_export_onnx()
    def forward(self, frames, **kwargs):
        if "is_tracing" in kwargs:
            frame_masks = frames[:, 3:4]
            frames = frames[:, :3]
        else:
            frame_masks = frames.mask.as_tensor()
            frames = frames.as_tensor()

        xs = self.body(frames)
        out: Dict[str, torch.Tensor] = {}
        for name, x in xs.items():
            n_mask = tvF.resize(frame_masks, x.shape[-2:])
            n_mask = n_mask.to(torch.bool)
            out[name] = (x, n_mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(self, name: str, train_backbone: bool, return_interm_layers: bool, dilation: bool, **kwargs):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(),
            norm_layer=FrozenBatchNorm2d,
        )
        num_channels = 512 if name in ("resnet18", "resnet34") else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers, **kwargs)


class Joiner(nn.Sequential):
    """A sequential wrapper for backbone and position embedding.

    `self.forward` returns a tuple:
        - list of feature maps from backbone
        - list of position encoded feature maps
    """

    def __init__(self, backbone: Backbone, position_embedding: nn.Module, tracing: bool = None):
        super().__init__(backbone, position_embedding)
        self.tracing = tracing

    @property
    def tracing(self):
        return self._tracing

    @tracing.setter
    def tracing(self, is_tracing: bool = None):
        self._tracing = is_tracing
        if is_tracing is not None:  # Update backbone tracing
            self[0].tracing = is_tracing
        else:  # Get same tracing property from backbone
            self._tracing = self[0].tracing

    @assert_and_export_onnx()
    def forward(self, frames: aloscene.Frame, **kwargs):

        xs = self[0](frames, **kwargs)
        out = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos_encoding = self[1](x).to(x[0].dtype)
            pos.append(pos_encoding)

        return out, pos


def build_backbone():
    position_embedding = build_position_encoding()
    train_backbone = True
    return_interm_layers = True  # Usefull if doing a mask
    dilation = False  # Usefull for DC5 in DETR
    backbone = Backbone("resnet50", train_backbone, return_interm_layers, dilation)

    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels

    return model


if __name__ == "__main__":

    backbone = build_backbone()
    print("backbone", backbone)

    pass
