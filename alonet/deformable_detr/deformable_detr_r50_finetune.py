import math
import torch
from torch import nn

from alonet.deformable_detr.deformable_detr import _get_clones
from alonet.deformable_detr import DeformableDetrR50, DeformableDetrR50Refinement


class DeformableDetrR50Finetune(DeformableDetrR50):
    """Pre made helpfull class to finetune the Deformable
    :mod:`Deformable DetrR50 <alonet.deformable_detr.deformable_detr_r50>` model on a custom class.

    Parameters
    ----------
    num_classes : int
        Number of classes to use
    activation_fn : str, optional
        Activation function to use in :attr:`class_embed` layer, by default "sigmoid"
    *args : Namespace
        Arguments used in :mod:`Deformable DetrR50 <alonet.deformable_detr.deformable_detr_r50>` module
    **kwargs : dict
        Aditional arguments used in :mod:`Deformable DetrR50 <alonet.deformable_detr.deformable_detr_r50>` module

    Raises
    ------
    Exception
        :attr:`activation_fn` must be "softmax" or "sigmoid". However, :attr:`activation_fn` = "softmax" implies to work with background class.
        That means increases in one the :attr:`num_classes` automatically.
    """

    def __init__(self, num_classes: int, *args, activation_fn: str = "sigmoid", **kwargs):
        """Init method"""
        if activation_fn not in ["sigmoid", "softmax"]:
            raise Exception(f"activation_fn = {activation_fn} must be one of this two values: 'sigmoid' or 'softmax'.")

        super().__init__(*args, **kwargs)

        self.activation_fn = activation_fn
        self.background_class = num_classes if self.activation_fn == "softmax" else None
        num_classes += 1 if self.activation_fn == "softmax" else 0  # Add bg class

        # Replace the class_embed layer a new layer once the deformable-detr-r50 weight are loaded
        self.class_embed = nn.Linear(self.transformer.d_model, num_classes)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        self.class_embed = self.class_embed.to(self.device)
        num_pred = self.transformer.decoder.num_layers
        self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])


class DeformableDetrR50RefinementFinetune(DeformableDetrR50Refinement):
    """
    Pre made helpfull class to finetune the
    :mod:`Deformable DetrR50 with refinement <alonet.deformable_detr.deformable_detr_r50_refinement>` model on a custom class.

    Parameters
    ----------
    num_classes : int
        Number of classes to use
    activation_fn : str, optional
        Activation function to use in :attr:`class_embed` layer, by default "sigmoid"
    *args : Namespace
        Arguments used in :mod:`Deformable DetrR50 with refinement <alonet.deformable_detr.deformable_detr_r50_refinement>` module
    **kwargs : dict
        Aditional arguments used in :mod:`Deformable DetrR50 with refinement <alonet.deformable_detr.deformable_detr_r50_refinement>` module

    Raises
    ------
    Exception
        :attr:`activation_fn` must be "softmax" or "sigmoid". However, :attr:`activation_fn` = "softmax" implies to work with background class.
        That means increases in one the :attr:`num_classes` automatically.
    """

    def __init__(self, num_classes: int, *args, activation_fn: str = "sigmoid", **kwargs):
        """Init method"""

        if activation_fn not in ["sigmoid", "softmax"]:
            raise Exception(f"activation_fn = {activation_fn} must be one of this two values: 'sigmoid' or 'softmax'.")

        super().__init__(*args, **kwargs)

        self.activation_fn = activation_fn
        self.background_class = num_classes if self.activation_fn == "softmax" else None
        num_classes += 1 if self.activation_fn == "softmax" else 0  # Add bg class

        # Replace the class_embed layer a new layer once the deformable-detr-r50 weight are loaded
        self.class_embed = nn.Linear(self.transformer.d_model, num_classes)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        self.class_embed = self.class_embed.to(self.device)
        num_pred = self.transformer.decoder.num_layers
        self.class_embed = _get_clones(self.class_embed, num_pred)


if __name__ == "__main__":
    # Setup a new Detr Model with 2 class
    # Additionally, we're gonna load the pretrained deformable detr-r50 weights.
    deformable_detr_r50_finetune = DeformableDetrR50Finetune(num_classes=2, weights="deformable-detr-r50")
    deformable_detr_r50_finetune = DeformableDetrR50RefinementFinetune(
        num_classes=2, weights="deformable-detr-r50-refinement"
    )
