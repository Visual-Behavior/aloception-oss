"""Module to create a custom :mod:`PanopticHead <alonet.detr_panoptic.detr_panoptic>` model using
:mod:`DeformableDetrR50 <alonet.deformable_detr.deformable_detr_r50>` as based model, which allows to upload a
decided pretrained weights and change the number of outputs in :attr:`class_embed` layer, in order to train custom
classes.
"""

import torch
import math
from argparse import Namespace

from alonet.deformable_detr_panoptic import DeformableDetrR50Panoptic
from alonet.common.weights import load_weights


class DeformableDetrR50PanopticFinetune(DeformableDetrR50Panoptic):
    """Pre made helpfull class to finetune the :mod:`DeformableDetrR50 <alonet.deformable_detr.deformable_detr_r50>`
    and use a pretrained :mod:`PanopticHead <alonet.detr_panoptic.detr_panoptic>`.

    Parameters
    ----------
    num_classes : int
        Number of classes in the :attr:`class_embed` output layer
    base_weights : str, optional
        Load weights to the original
        :mod:`DeformableDetrR50Panoptic <alonet.deformable_detr_panoptic.deformable_detr_r50_panoptic>`,
        by default "deformable-detr-r50-panoptic"
    weights : str, optional
        Weights for finetune model, by default None
    use_bn_layers : bool, optional
        Replace group norm layer in :mod:`PanopticHead <alonet.detr_panoptic.detr_panoptic>` by batch norm layer,
        by default False
    **kwargs
        Initial parameters of
        :mod:`DeformableDetrR50panoptic <alonet.deformable_detr_panoptic.deformable_detr_r50_panoptic>` module

    Raises
    ------
    ValueError
        :attr:`weights` must be a '.pth' or '.ckpt' file
    """

    def __init__(
        self,
        num_classes: int,
        base_weights: str = "deformable-detr-r50-panoptic",
        weights: str = None,
        use_bn_layers: bool = False,
        *args: Namespace,
        **kwargs: dict,
    ):
        """Init method"""
        super().__init__(*args, weights=base_weights, **kwargs)

        self.detr.background_class = num_classes if self.detr.activation_fn == "softmax" else None
        num_classes += 1 if self.detr.activation_fn == "softmax" else 0  # Add bg class

        # Replace the class_embed layer a new layer once the deformable-detr-r50 weight are loaded
        self.detr.class_embed = torch.nn.Linear(self.detr.transformer.d_model, num_classes)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.detr.class_embed.bias.data = torch.ones(num_classes) * bias_value
        self.detr.class_embed = self.detr.class_embed.to(self.device)
        num_pred = self.detr.transformer.decoder.num_layers
        self.detr.class_embed = torch.nn.ModuleList([self.detr.class_embed for _ in range(num_pred)])

        # Replace group by batch norm layers
        if use_bn_layers:
            for ignl in range(1, 6):
                gname = "gn" + str(ignl)
                glayer = getattr(self.mask_head, gname)
                setattr(self.mask_head, gname, torch.nn.BatchNorm2d(glayer.num_channels))

        # Load weights procedure
        if weights is not None:
            if ".pth" in weights or ".ckpt" in weights:
                load_weights(self, weights, self.device)
            else:
                raise ValueError(f"Unknown weights: '{weights}'")


if __name__ == "__main__":
    # Setup a new Detr Model with 2 class and the background class equal to 0.
    # Additionally, we're gonna load the pretrained deformable-detr-r50 weights.
    panoptic_finetune = DeformableDetrR50PanopticFinetune(num_classes=2)
