"""Module to create a custom :mod:`PanopticHead <alonet.detr_panoptic.detr_panoptic>` model using
:mod:`DetrR50 <alonet.detr.detr_r50>` as based model, which allows to upload a decided pretrained weights and
change the number of outputs in :attr:`class_embed` layer, in order to train custom classes.
"""

from torch import nn
from argparse import Namespace
from alonet.detr_panoptic import PanopticHead
from alonet.detr import DetrR50
from alonet.common.weights import load_weights


class DetrR50PanopticFinetune(PanopticHead):
    """Pre made helpfull class to finetune the :mod:`DetrR50 <alonet.detr.detr_r50>` and use a pretrained
    :mod:`PanopticHead <alonet.detr_panoptic.detr_panoptic>`.

    Parameters
    ----------
    num_classes : int
        Number of classes in the :attr:`class_embed` output layer
    background_class : int, optional
        Background class, by default None
    base_model : torch.nn, optional
        Base model to couple PanopticHead, by default :mod:`DetrR50 <alonet.detr.detr_r50>`
    base_weights : str, optional
        Load weights from original :mod:`DetrR50 <alonet.detr.detr_r50>` +
        :mod:`PanopticHead <alonet.detr_panoptic.detr_panoptic>`,
        by default "/home/johan/.aloception/weights/detr-r50-panoptic/detr-r50-panoptic.pth"
    freeze_detr : bool, optional
        Freeze :mod:`DetrR50 <alonet.detr.detr_r50>` weights, by default False
    weights : str, optional
        Weights for finetune model, by default None

    Raises
    ------
    ValueError
        :attr:`weights` must be a '.pth' or '.ckpt' file
    """

    def __init__(
        self,
        num_classes: int,
        background_class: int = None,
        base_model: nn = None,
        base_weights: str = "detr-r50-panoptic",
        freeze_detr: bool = False,
        weights: str = None,
        *args: Namespace,
        **kwargs: dict,
    ):
        """Init method"""
        base_model = base_model or DetrR50(*args, background_class=background_class, num_classes=250, **kwargs)
        super().__init__(*args, DETR_module=base_model, freeze_detr=freeze_detr, weights=base_weights, **kwargs)

        self.detr.num_classes = num_classes
        # Replace the class_embed layer a new layer once the detr-r50 weight are loaded
        # + 1 to include the background class.
        self.detr.background_class = self.detr.num_classes if background_class is None else background_class
        self.detr.num_classes = num_classes + 1
        self.detr.class_embed = nn.Linear(self.detr.hidden_dim, self.detr.num_classes)
        self.detr.class_embed = self.detr.class_embed.to(self.device)

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
