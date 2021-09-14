import numpy as np
import cv2
from numpy.lib.arraysetops import isin
import torch

from typing import Union
from torch import Tensor

import aloscene
from aloscene.renderer import View
from aloscene.io.mask import load_mask
from aloscene.labels import Labels


class Mask(aloscene.tensors.SpatialAugmentedTensor):
    """Binary or Float Mask

    Parameters
    ----------
    x :
        path to the mask file (png) or tensor (values between 0. and 1.)
    """

    @staticmethod
    def __new__(cls, x, labels: Union[dict, Labels] = None, *args, **kwargs):
        # Load frame from path
        if isinstance(x, str):
            x = load_mask(x)
            kwargs["names"] = ("N", "H", "W")
        tensor = super().__new__(cls, x, *args, **kwargs)
        tensor.add_label("labels", labels, align_dim=["N"], mergeable=False)
        return tensor

    def __init__(self, x, *args, **kwargs):
        super().__init__(x)

    def append_labels(self, labels: Labels, name: str = None):
        """Attach a set of labels to the masks.

        Parameters
        ----------
        labels: aloscene.Labels
            Set of labels to attached to the masks
        name: str
            If none, the label will be attached without name (if possible). Otherwise if no other unnamed
            labels are attached to the frame, the labels will be added to the set of labels.
        """
        self._append_label("labels", labels, name)

    GLOBAL_COLOR_SET = np.random.uniform(0, 1, (300, 3))

    def __get_view__(self, title=None):
        """Create a view of the frame"""
        assert self.names[0] != "T" and self.names[1] != "B"
        frame = self.cpu().rename(None).permute([1, 2, 0]).detach().contiguous().numpy()

        # If it does not mask, raise Error
        assert len(self) > 0, "Len(Mask) must be higher than 0"

        # Try to retrieve the associated label ID (if any)
        labels = self.labels if isinstance(self.labels, aloscene.Labels) else [None] * len(self)
        annotations = []
        if isinstance(self.labels, aloscene.Labels):
            assert self.labels.encoding == "id"

        if frame.shape[-1] != 1:
            frame = np.concatenate([np.zeros_like(frame[..., [0]]), frame], axis=-1)  # Add background class as 0
            frame = np.argmax(frame, axis=-1).astype("int")  # Get one mask by ID

            assert len(labels) == len(self)  # Required to plot labels
            for i, label in enumerate(labels):  # Add ID in text and use same color by object ID
                if label is not None:
                    # Change ID if labels are defined
                    label = int(label)
                    frame[frame == i + 1] = (label + 1) % len(self.GLOBAL_COLOR_SET)

                    # Get mass center to put text in frame
                    feat = self[i].cpu().detach().contiguous().numpy()  # Get i_mask
                    mass_y, mass_x = np.where(feat > 0.5)
                    x, y = np.average(mass_x), np.average(mass_y)
                    color = self.GLOBAL_COLOR_SET[(label + 1) % len(self.GLOBAL_COLOR_SET)]
                    color = (0, 0, 0)
                    annotations.append({"color": color, "x": int(x), "y": int(y), "text": labels.labels_names[label]})

            # Frame construction by segmentation masks
            frame = self.GLOBAL_COLOR_SET[frame]
        elif labels[0] is not None:
            label = int(labels[0])

            # Get mass center to put text in frame
            mass_y, mass_x = np.where(frame[..., 0] > 0.5)
            x, y = np.average(mass_x), np.average(mass_y)
            color = (0, 0, 0)
            text = labels.labels_names[label] if "labels_names" in labels else str(label)
            annotations.append({"color": color, "x": int(x), "y": int(y), "text": text})

            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        # Add relative text in frame
        for anno in annotations:
            cv2.putText(
                frame,
                anno["text"],
                (anno["x"], anno["y"]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                anno["color"],
                1,
                cv2.LINE_AA,
            )
        return View(frame, title=title)

    def get_view(self, frame: Tensor = None, size: tuple = None, labels_set: str = None, **kwargs):
        from aloscene import Frame

        if not isinstance(self.labels, aloscene.Labels):
            return super().get_view(size=size, frame=frame, **kwargs)

        if frame is not None:
            if len(frame.shape) > 3:
                raise Exception(f"Expect image of shape c,h,w. Found image with shape {frame.shape}")
            assert isinstance(frame, Frame)
        else:
            size = self.shape[1:]
            frame = torch.zeros(3, *size)
            frame = Frame(frame, names=("C", "H", "W"), normalization="01")

        masks = self.__get_view__(**kwargs).image
        frame = frame.norm01().cpu().rename(None).permute([1, 2, 0]).detach().contiguous().numpy()
        frame = cv2.resize(frame, (self.shape[-1], self.shape[-2]))
        frame = 0.4 * frame + 0.6 * masks
        return View(frame, **kwargs)
