"""Module to create a custom :mod:`PanopticHead <alonet.detr_panoptic.detr_panoptic>` model using
:mod:`DetrR50 <alonet.detr.detr_r50>` as based model, which allows to upload a decided pretrained weights and
change the number of outputs in :attr:`class_embed` layer, in order to train custom classes.
"""

import torch
from argparse import Namespace

from alonet.detr_panoptic import DetrR50Panoptic
from alonet.common.weights import load_weights


class DetrR50PanopticFinetune(DetrR50Panoptic):
    """Pre made helpfull class to finetune the :mod:`DetrR50 <alonet.detr.detr_r50>` and use a pretrained
    :mod:`PanopticHead <alonet.detr_panoptic.detr_panoptic>`.

    Parameters
    ----------
    num_classes : int
        Number of classes in the :attr:`class_embed` output layer
    background_class : int, optional
        Background class, by default None
    base_weights : str, optional
        Load weights to the original :mod:`DetrR50Panoptic <alonet.detr_panoptic.detr_r50_panoptic>`,
        by default "detr-r50-panoptic"
    weights : str, optional
        Load weights from path, by default None
    use_bn_layers : bool, optional
        Replace group norm layer in :mod:`PanopticHead <alonet.detr_panoptic.detr_panoptic>` by batch norm layer,
        by default False
    **kwargs
        Initial parameters of :mod:`DetrR50panoptic <alonet.detr_panoptic.detr_r50_panoptic>` module

    Raises
    ------
    ValueError
        :attr:`weights` must be a '.pth' or '.ckpt' file
    """

    def __init__(
        self,
        num_classes: int,
        background_class: int = None,
        base_weights: str = "detr-r50-panoptic",
        weights: str = None,
        use_bn_layers: bool = False,
        *args: Namespace,
        **kwargs: dict,
    ):
        """Init method"""
        super().__init__(*args, weights=base_weights, **kwargs)

        self.detr.num_classes = num_classes
        # Replace the class_embed layer a new layer once the detr-r50 weight are loaded
        # + 1 to include the background class.
        self.detr.background_class = self.detr.num_classes if background_class is None else background_class
        self.detr.num_classes = num_classes + 1
        self.detr.class_embed = torch.nn.Linear(self.detr.hidden_dim, self.detr.num_classes)
        self.detr.class_embed = self.detr.class_embed.to(self.device)

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
    # Additionally, we're gonna load the pretrained detr-r50 weights.
    panoptic_finetune = DetrR50PanopticFinetune(num_classes=2)
