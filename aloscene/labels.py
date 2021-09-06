import torch
from torch import Tensor
import torchvision

from typing import *
import numpy as np
import cv2

import aloscene


class Labels(aloscene.tensors.AugmentedTensor):
    """Boxes2D Tensor."""

    @staticmethod
    def __new__(
        cls,
        x,
        encoding: str = "one_hot",
        labels_names: list = None,
        background_class: int = None,
        scores: torch.Tensor = None,
        names=("N"),
        *args,
        **kwargs,
    ):
        """The Labels Augmented Tensor class car be used for classification and
        object detection.

        Parameters
        ----------
        x: Any data
            Data for the tensor
        encoding: str
            How the labels are encoded, can be one of "one_hot" or "id"
        labels_names: list | None
            Can be None, otherwise this is the class associatd with each label ID
        background_class: int | None
        scores: torch.tensor }
        """
        tensor = super().__new__(cls, x, *args, names=names, **kwargs)

        # Encoding
        if encoding not in ["one_hot", "id"]:
            raise Exception(f"Passed encoding {encoding} not handle")
        tensor.add_property("encoding", encoding)

        # Labels info
        tensor.add_property("labels_names", labels_names)
        tensor.add_property("background_class", background_class)

        if scores is not None:
            assert scores.shape == tensor.shape
        tensor.add_label("scores", scores, align_dim=["N"], mergeable=True)

        return tensor

    def __init__(self, x, *args, **kwargs):
        super().__init__(x)

    def _hflip(self, *args, **kwargs):
        return self

    def _resize(self, *args, **kwargs):
        return self

    def _crop(self, *args, **kwargs):
        return self

    def _pad(self, *args, **kwargs):
        return self

    def _spatial_shift(self, shift_y: float, shift_x: float, **kwargs):
        return self
