import torch
from torch import Tensor
import torchvision

from typing import *
import numpy as np
import cv2

import aloscene
from aloscene.renderer import View, put_adapative_cv2_text, adapt_text_size_to_frame


class Labels(aloscene.tensors.AugmentedTensor):
    """Boxes2D Tensor."""

    @staticmethod
    def __new__(
        cls,
        x,
        encoding: str = "id",
        labels_names: Union[list, None] = None,
        scores: Union[torch.Tensor, None] = None,
        names=None,
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
            How the labels are encoded, can be one of "one_hot" or "id". Default is "id", therefore the default names
            will be set to ("N",). IF "one_hot", the default names will be ("N", None) with the last dimension being
            the number of possible class.
        labels_names: list | None
            Can be None, otherwise this is the class names associatd with each of the possible label id.
            By default ("N",) will be used for encoding="id". ("N", None) will be used for one_hot encoding.
        scores: torch.tensor
            Scores for each of the N provided labels.
        """
        if names is None and encoding == "id":
            names = ("N",)
        elif names is None and encoding == "one_hot":
            names = ("N", None)

        if encoding == "one_hot":
            raise Exception("One hot not yet supported (todo)")

        tensor = super().__new__(cls, x, *args, names=names, **kwargs)
        if len(tensor.shape) == 0 and encoding == "id":
            raise Exception("aloscene.Labels must be at least 1 dimensional (N,) with encoding=id.")
        elif len(tensor.shape) == 1 and encoding == "one_hot":
            raise Exception("aloscene.Labels must be at least 2 dimensional (N, None) with encoding=`one_hot`")

        # Encoding
        if encoding not in ["one_hot", "id"]:
            raise Exception(f"Passed encoding {encoding} not handle")

        tensor.add_property("encoding", encoding)
        tensor.add_property("labels_names", labels_names)

        if scores is not None:
            assert scores.shape == tensor.shape
        tensor.add_child("scores", scores, align_dim=["N"], mergeable=True)

        return tensor

    def __init__(self, x, *args, **kwargs):
        super().__init__(x)

    def get_view(self, frame: Union[Tensor, None] = None, **kwargs):
        """Create a view of the boxes a frame

        Parameters
        ----------
        frame: aloscene.Frame
            Tensor of type Frame to display the boxes on. If the frame is None, nothing wll be returned.
        """
        if frame is not None:
            frame_size = (frame.H, frame.W)
            x0, y0 = int(frame_size[0] * 0.10), int(frame_size[1] * 0.10)
            size = adapt_text_size_to_frame(1.0, frame_size)

            if frame_size[0] < 200 or frame_size[0] < 200:
                print("Warning: frame label does not get render since the frame is too small (< 200 px)")
                return None

            # Get an imave with values between 0 and 1
            frame = frame.norm01().cpu().rename(None).permute([1, 2, 0]).detach().contiguous().numpy()

            for label in self:

                if self.labels_names is None:
                    text = str(int(label))
                else:
                    label_id = int(label)
                    if label_id < 0 or label_id >= len(self.labels_names):
                        text = str(int(label))
                    else:
                        text = self.labels_names[int(label)]

                size = 1 * ((frame_size[0] + frame_size[0]) / 2) / 1000

                put_adapative_cv2_text(frame, frame_size, text, x0, y0)

                c_size = size * 20
                y0 += (c_size * 2) + 4

            return View(frame, **kwargs)
        return None

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
