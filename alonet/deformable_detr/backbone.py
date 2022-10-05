"""
Backbone modules for Deformable DETR
"""

import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter


from alonet.detr.backbone import BackboneBase as DetrBackboneBase
from alonet.detr.backbone import Joiner as DetrJoiner
from alonet.detr.backbone import is_main_process, FrozenBatchNorm2d


class BackboneBase(DetrBackboneBase):
    """Base class to define behavior of backbone"""

    def __init__(self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool, **kwargs):
        super().__init__(
            backbone,
            train_backbone,
            num_channels=2048,  # Don't care. Will be override
            return_interm_layers=True,  # Don't care. Will be override
            **kwargs,
        )
        if return_interm_layers:
            # Ignore layer1 in forward. Use layer1 for panoptic purposes
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            # return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [4, 8, 16, 32]
            self.num_channels = [256, 512, 1024, 2048]
        else:
            return_layers = {"layer4": "0"}
            self.strides = [32]
            self.num_channels = [2048]
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(self, name: str, train_backbone: bool, return_interm_layers: bool, dilation: bool, **kwargs):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(),
            norm_layer=FrozenBatchNorm2d,
        )
        assert name not in ("resnet18", "resnet34"), "number of channels are hard coded"
        super().__init__(backbone, train_backbone, return_interm_layers, **kwargs)


class Joiner(DetrJoiner):
    """A sequential wrapper for backbone and position embedding.

    `self.forward` returns a tuple:
        - list of feature maps from backbone
        - list of position encoded feature maps
    """

    def __init__(self, backbone, position_embedding, tracing: bool = None):
        super().__init__(backbone, position_embedding, tracing)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels
