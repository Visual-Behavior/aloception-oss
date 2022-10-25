"""Binary or float Mask, use in oclussion, segmentation, crop and forward process"""
import numpy as np
import cv2
import torch

from typing import Union
from torch import Tensor

import aloscene
from aloscene.renderer import View
from aloscene.io.mask import load_mask
from aloscene.labels import Labels
import torchvision.transforms.functional as F


class Mask(aloscene.tensors.SpatialAugmentedTensor):
    """
    Parameters
    ----------
    x : Union[torch.Tensor, str]
        Path to the mask file (png) or tensor (values between 0. and 1.)
    labels : Union[dict, :mod:`Labels <aloscene.labels>`], optional
        Labels for each mask, used for rendering segmentation maps
    """

    @staticmethod
    def __new__(cls, x, labels: Union[dict, Labels, None] = None, *args, **kwargs):
        # Load frame from path
        if isinstance(x, str):
            x = load_mask(x)
            kwargs["names"] = ("N", "H", "W")
        tensor = super().__new__(cls, x, *args, **kwargs)
        tensor.add_child("labels", labels, align_dim=["N"], mergeable=True)
        return tensor

    def append_labels(self, labels: Labels, name: Union[str, None] = None):
        """Attach a set of labels to the masks.

        Parameters
        ----------
        labels : :mod:`Labels <aloscene.labels>`
            Set of labels to attached to the masks
        name : str
            If none, the label will be attached without name (if possible). Otherwise if no other unnamed
            labels are attached to the frame, the labels will be added to the set of labels.
        """
        self._append_child("labels", labels, name)

    def iou_with(self, mask2) -> torch.Tensor:
        """IoU calculation between mask2 and itself

        Parameters
        ----------
        mask2 : :mod:`Mask <aloscene.mask>`
            Masks with size (M,H,W)

        Returns
        -------
        torch.Tensor
            IoU matrix of size (N,M)

        Raises
        ------
        Exception
            Features size (H,W) between masks have to be the same
        """
        if len(self) == 0 and len(mask2) == 0:
            return torch.rand(0, 0)
        elif len(self) == 0:
            return torch.rand(0, len(mask2))
        elif len(mask2) == 0:
            return torch.rand(len(self), 0)
        mask1 = self.flatten(["H", "W"], "features").rename(None)  # Binary mask (N, f=WxH)
        mask2 = mask2.flatten(["H", "W"], "features").rename(None)  # Binary mask (M, f=WxH)
        assert mask1.shape[-1] == mask2.shape[-1]
        intersection = mask1.matmul(mask2.transpose(0, 1))  # (N, M)
        mask1, mask2 = mask1.sum(-1, keepdim=True), mask2.sum(-1, keepdim=True)
        union = mask1.repeat(1, len(mask2)) + mask2.transpose(0, 1)  # (N, M)
        union[union == 0] = 0.001  # Avoid divide by 0
        return intersection / (union - intersection)

    def get_view(
        self,
        frame: Union[Tensor, None] = None,
        size: Union[tuple, None] = None,
        labels_set: Union[str, None] = None,
        color_by_cat: bool = True,
        **kwargs,
    ):
        """Get view of segmentation mask and used it in a input :mod:`Frame <aloscene.frame>`

        Parameters
        ----------
        frame : Tensor, optional
            Frame where the segmentation mask will be displayed, by default None
        size : tuple, optional
            Size of a desired masks, by default not-resize
        labels_set : str, optional
            Set of labels to show in segmentation when multiple labels are defined, by default None
        color_by_cat : bool, optional
            Set same color by category ID, by default True

        Returns
        -------
        Renderer.View
            Frame view, ready to be rendered

        Raises
        ------
        Exception
            Input frame must be a :mod:`Frame <aloscene.frame>` object
        """
        from aloscene import Frame

        if not (hasattr(self, "labels") and isinstance(self.labels, (aloscene.Labels, dict))):
            return super().get_view(size=size, frame=frame, **kwargs)

        if frame is not None:
            if len(frame.shape) > 3:
                raise Exception(f"Expect image of shape c,h,w. Found image with shape {frame.shape}")
            assert isinstance(frame, Frame)
        else:
            size = self.shape[1:]
            frame = torch.zeros(3, *size)
            frame = Frame(frame, names=("C", "H", "W"), normalization="01")

        masks = self.__get_view__(labels_set=labels_set, color_by_cat=color_by_cat, **kwargs).image

        frame = frame.norm01().cpu().rename(None).permute([1, 2, 0]).detach().contiguous().numpy()
        frame = cv2.resize(frame, (self.shape[-1], self.shape[-2]))
        if masks.shape[-1] > 0:
            frame = 0.2 * frame + 0.8 * masks
        return View(frame, **kwargs)

    def __get_view__(
        self, labels_set: Union[str, None] = None, title: Union[str, None] = None, color_by_cat: bool = True, **kwargs
    ):
        """Create a view of the frame"""
        from alodataset.utils.panoptic_utils import id2rgb

        frame, annotations = self.mask2id(labels_set=labels_set, return_ann=True, return_cats=color_by_cat)

        # Frame construction by segmentation masks
        if hasattr(self, "labels") and self.labels is not None and len(self) > 0:
            frame = id2rgb(frame)

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

    def mask2id(self, labels_set: Union[str, None] = None, return_ann: bool = False, return_cats: bool = False):
        """Create a panoptic view of the frame, where each pixel represent one class

        Parameters
        ----------
        labels_set : str, optional
            If multilabels are handled, get mask_id by a set of label desired, by default None
        return_ann : bool, optional
            Return annotations to get_view function, by default False
        return_cats : bool, optional
            Return categories ID instance ID, by default False.

        Returns
        -------
        np.array
            Array of (H,W) dimensions, where each value represent one class
        """
        from alodataset.utils.panoptic_utils import VOID_CLASS_ID

        assert self.names[0] != "T" and self.names[1] != "B"
        frame = self.cpu().rename(None).permute([1, 2, 0]).detach().contiguous().numpy()

        # Try to retrieve the associated label ID (if any)
        labels = self._get_set_children(labels_set=labels_set)
        annotations = []
        if hasattr(self, "labels") and self.labels is not None and len(labels) > 0:
            assert len(labels) == len(self)  # Required to make panoptic view

            frame = np.concatenate([np.zeros_like(frame[..., [0]]), frame], axis=-1)  # Add BG class with ID=VOID
            frame = np.argmax(frame, axis=-1).astype("int") + VOID_CLASS_ID  # Get one mask by ID
            copy_frame = frame.copy()

            for i, label in enumerate(labels):  # Add ID in text and use same color by object ID
                # Change ID if labels are defined
                if label is not None:
                    label = int(label)

                    if return_cats:
                        frame[copy_frame == (i + VOID_CLASS_ID + 1)] = label

                    if return_ann:
                        feat = self[i].cpu().detach().contiguous().numpy()  # Get i_mask
                        mass_y, mass_x = np.where(feat > 0.5)
                        x, y = np.average(mass_x), np.average(mass_y)
                        x = 0 if np.isnan(x) else x
                        y = 0 if np.isnan(y) else y
                        text = str(label) if labels.labels_names is None else labels.labels_names[label]
                        annotations.append({"color": (0, 0, 0), "x": int(x), "y": int(y), "text": text})
        if return_ann:
            return frame, annotations
        return frame

    def _get_set_children(self, labels_set: Union[str, None] = None):
        if not (labels_set is None or isinstance(self.labels, dict)):
            raise Exception(
                f"Trying to display a set of labels ({labels_set}) while masks do not have multiple set of labels"
            )
        elif labels_set is not None and isinstance(self.labels, dict) and labels_set not in self.labels:
            raise Exception(
                f"Trying to display a set of labels ({labels_set}) while masks not have it. Available set: ("
                + f"{[key for key in self.labels]}"
                + ") "
            )
        elif not hasattr(self, "labels"):
            labels = [None] * len(self)
        elif labels_set is not None and isinstance(self.labels, dict):
            labels = self.labels[labels_set]
            assert isinstance(labels, aloscene.Labels) and labels.encoding == "id"
        elif isinstance(self.labels, aloscene.Labels):
            labels = self.labels
            assert labels.encoding == "id"
        else:
            labels = [None] * len(self)
        return labels


    def _spatial_shift(self, shift_y: float, shift_x: float, **kwargs):
            """
            Spatially shift the Mask.
            Parameters
            ----------
            shift_y: float
                Shift percentage on the y axis. Could be negative or positive
            shift_x: float
                Shift percentage on the x axis. Could ne negative or positive.
            Returns
            -------
            shifted_tensor: aloscene.AugmentedTensor
                shifted tensor
            """
            n_frame = self.clone()

            if "N"  in self.names:
                ind_name="N"
            elif "C" in self.names:
                ind_name="C"


            n_chans=self.shape[self.names.index(ind_name) ]


            x_shift = int(shift_x * self.W)
            y_shift = int(shift_y * self.H)


            frame_data = n_frame.as_tensor()

            permute_idx = list(range(0, len(self.shape)))
            last_current_idx = permute_idx[-1]
            permute_idx[-1] = permute_idx[self.names.index(ind_name)]
            permute_idx[self.names.index(ind_name)] = last_current_idx

            # n_frame_mean = frame_data.permute(permute_idx)
            # n_frame_mean = n_frame_mean.flatten(end_dim=-2)
            n_frame_mean = torch.tensor([1])
            n_shape = [1] * len(self.shape)
            n_shape[self.names.index(ind_name)] = n_chans



            frame_data = torch.roll(frame_data, x_shift, dims=self.names.index("W"))
            # Fillup the shifted area with the mean

            if x_shift >= 1:
                frame_data[self.get_slices({"W": slice(0, x_shift)})] = n_frame_mean
            elif x_shift <= -1:
                frame_data[self.get_slices({"W": slice(x_shift, -1)})] = n_frame_mean

            frame_data = torch.roll(frame_data, y_shift, dims=self.names.index("H"))

            if y_shift >= 1:
                frame_data[self.get_slices({"H": slice(0, y_shift)})] = n_frame_mean
            elif y_shift <= -1:
                frame_data[self.get_slices({"H": slice(y_shift, -1)})] = n_frame_mean

            n_frame.data = frame_data

            return n_frame

    def _rotate(self, angle, **kwargs):
        """Rotate the mask and fill the areas outside with 1 to show that these pixels become invalid

        Parameters
        ----------
        angle : float

        Returns
        -------
        rotated : aloscene.SpatialAugmentedTensor
            rotated tensor
        """
        # If a SpatialAgumentedTensor is empty, rotate operation does not work. Use view instead.
        assert not (
            ("N" in self.names and self.size("N") == 0) or ("C" in self.names and self.size("C") == 0)
        ), "rotation is not possible on an empty tensor"
        if len(self.shape)==3:
            self=self.temporal(0)

        new_self=self.type(torch.LongTensor)
        print("dtype",self.dtype)

        new_mask=F.rotate(new_self.rename(None), angle,fill=1.0)
        return new_mask.reset_names()
