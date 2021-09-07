import torch
import numpy as np
import cv2

from typing import Union
from torch import Tensor

import aloscene
from aloscene.renderer import View
from aloscene.labels import Labels


class Panoptic(aloscene.tensors.AugmentedTensor):
    """Binary mask for each label to represent panoptic segmentations
    """

    @staticmethod
    def __new__(cls, x: Tensor, labels: Union[dict, Labels] = None, names=("N", "H", "W"), *args, **kwargs):
        # Load frame from path
        tensor = super().__new__(cls, x, names=names, *args, **kwargs)
        tensor.add_label("labels", labels, align_dim=["N"], mergeable=True)
        tensor.add_property("frame_size", x.shape[1:])
        return tensor

    def __init__(self, x, *args, **kwargs):
        super().__init__(x)

    def append_labels(self, labels: Labels, name: str = None):
        """Attach a set of labels to the boxes.

        Parameters
        ----------
        labels: aloscene.Labels
            Set of labels to attached to the frame
        name: str
            If none, the label will be attached without name (if possible). Otherwise if no other unnamed
            labels are attached to the frame, the labels will be added to the set of labels.
        """
        self._append_label("labels", labels, name)

    GLOBAL_COLOR_SET = np.random.uniform(0, 1, (300, 3))

    def get_view(self, frame: Tensor = None, size: tuple = None, labels_set: str = None, **kwargs):
        """Create a view of the boxes a frame

        Parameters
        ----------
        frame: aloscene.Frame
            Tensor of type Frame to display the boxes on. If the frameis None, a frame will be create on the fly.
        size: (tuple)
            (height, width) Desired size of the view. None by default
        """
        from aloscene import Frame

        size = size if size is not None else self.frame_size
        if frame is not None:
            if len(frame.shape) > 3:
                raise Exception(f"Expect image of shape c,h,w. Found image with shape {frame.shape}")
            assert isinstance(frame, Frame)
        else:
            size = self.frame_size
            frame = torch.zeros(3, *size)
            frame = Frame(frame, names=("C", "H", "W"), normalization="01")

        # Get an image with values between 0 and 1
        frame = frame.norm01().cpu().rename(None).permute([1, 2, 0]).detach().contiguous().numpy()
        frame = cv2.resize(frame, (self.size(2), self.size(1)), interpolation=cv2.INTER_NEAREST)

        # Get mask by index
        mask = self.cpu().detach().numpy()
        mask = np.concatenate([np.zeros_like(mask[[0]]), mask])  # Add background class as 0
        mask = np.argmax(mask, axis=0).astype("int")

        # Try to retrieve the associated label ID (if any)
        labels = self.labels if isinstance(self.labels, aloscene.Labels) else [None] * len(self)
        if isinstance(self.labels, aloscene.Labels):
            assert self.labels.encoding == "id"

        for i, label in enumerate(labels):
            if label is not None:  # First color to background
                mask[mask == i + 1] = int(label) + 1 % len(self.GLOBAL_COLOR_SET)

        # Add panotic segmentation to frame
        frame = 0.4 * frame + 0.6 * self.GLOBAL_COLOR_SET[mask]

        # Add id text
        for i, label in enumerate(labels):
            feat = self[i].cpu().numpy()
            mass_y, mass_x = np.where(feat > 0.5)
            x, y = np.average(mass_x), np.average(mass_y)
            if label is not None:
                label = int(label)
                color = self.GLOBAL_COLOR_SET[label + 1 % len(self.GLOBAL_COLOR_SET)]
                cv2.putText(frame, str(label), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # Resize frame
        frame = cv2.resize(frame, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)

        # Return the view to display
        return View(frame, **kwargs)
