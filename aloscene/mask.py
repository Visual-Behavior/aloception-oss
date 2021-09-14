import numpy as np
import cv2
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
        tensor.add_label("labels", labels, align_dim=["N"], mergeable=True)
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

        # Try to retrieve the associated label ID (if any)
        labels = self.labels if isinstance(self.labels, aloscene.Labels) else [None] * len(self)
        annotations = []
        if isinstance(self.labels, aloscene.Labels) and len(self) > 0:
            assert self.labels.encoding == "id"

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
                    text = str(label) if labels.labels_names is None else labels.labels_names[label]
                    annotations.append({"color": color, "x": int(x), "y": int(y), "text": text})

            # Frame construction by segmentation masks
            frame = self.GLOBAL_COLOR_SET[frame]

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

    def masks2panoptic(self):
        """Create a panoptic view of the frame, where each pixel represent one class

        Returns
        -------
        np.array
            Array of (H,W) dimensions, where each value represent one class
        """
        """"""
        assert self.names[0] != "T" and self.names[1] != "B"
        frame = self.cpu().rename(None).permute([1, 2, 0]).detach().contiguous().numpy()

        # Try to retrieve the associated label ID (if any)
        labels = self.labels if isinstance(self.labels, aloscene.Labels) else [None] * len(self)
        if isinstance(self.labels, aloscene.Labels) and len(self) > 0:
            assert self.labels.encoding == "id"

            frame = np.concatenate([np.zeros_like(frame[..., [0]]), frame], axis=-1)  # Add background class with ID=-1
            frame = np.argmax(frame, axis=-1).astype("int") - 1  # Get one mask by ID

            assert len(labels) == len(self)  # Required to plot labels
            for i, label in enumerate(labels):  # Add ID in text and use same color by object ID
                if label is not None:
                    # Change ID if labels are defined
                    label = int(label)
                    frame[frame == i] = label
        return frame

    def get_view(self, frame: Tensor = None, size: tuple = None, labels_set: str = None, **kwargs):
        """Get view of segmentation mask and used it in a input Frame

        Parameters
        ----------
        frame : Tensor, optional
            Frame where the segmentation mask will be displayed, by default None
        size : tuple, optional
            Size of a desired masks, by default not-resize
        labels_set : str, optional
            TODO set of labels to show in segmentation, by default all

        Returns
        -------
        Renderer.View
            Frame view, ready to render

        Raises
        ------
        Exception
            Input frame must be a aloscene.Frame object
        """
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
        if masks.shape[-1] > 0:
            frame = 0.4 * frame + 0.6 * masks
        return View(frame, **kwargs)

    def iou_with(self, mask2) -> torch.Tensor:
        """ IoU calculation between mask2 and itself

        Parameters
        ----------
        mask2 : aloscene.Mask
            Masks with size (M,H,W)

        Returns
        -------
        torch.Tensor
            IoU matrix of size (N,M)
        """
        mask1 = self.rename(None).view(len(self), -1)  # (N, WxH)
        mask2 = mask2.rename(None).view(len(mask2), -1)  # (M, WxH)
        intersection = mask1.matmul(mask2.transpose(0, 1))  # (N, M)
        union = torch.stack([mask1] * len(mask2), dim=1) + mask2  # (N, M, WxH)
        return intersection / union.sum(-1)  # (N, M)
