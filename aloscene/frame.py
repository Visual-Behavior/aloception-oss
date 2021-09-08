import torch
import torchvision
from typing import TypeVar, Union

# from collections import namedtuple

import aloscene
from aloscene.renderer import View
from aloscene.disparity import Disparity
from aloscene import BoundingBoxes2D, BoundingBoxes3D, Flow, Mask

# from aloscene.camera_calib import CameraExtrinsic, CameraIntrinsic
from aloscene.io.image import load_image

Frame = TypeVar("Frame")


class Frame(aloscene.tensors.SpatialAugmentedTensor):
    """Frame (Image) with associated labels and parameters"""

    @staticmethod
    def __new__(
        cls,
        x,
        boxes2d: Union[dict, BoundingBoxes2D] = None,
        boxes3d: Union[dict, BoundingBoxes3D] = None,
        flow: Flow = None,
        mask: Mask = None,
        segmentation: Mask = None,
        disparity: Disparity = None,
        normalization="255",
        mean_std=None,
        *args,
        **kwargs,
    ):
        """Frame (Image) with associated labels and parameters"""
        if isinstance(x, str):
            # Load frame from path
            x = load_image(x)
            normalization = "255"
            kwargs["names"] = ("C", "H", "W")
        tensor = super().__new__(cls, x, *args, **kwargs)

        # Add label
        tensor.add_label("boxes2d", boxes2d, align_dim=["B", "T"], mergeable=False)
        tensor.add_label("boxes3d", boxes3d, align_dim=["B", "T"], mergeable=False)
        tensor.add_label("flow", flow, align_dim=["B", "T"], mergeable=False)
        tensor.add_label("mask", mask, align_dim=["B", "T"], mergeable=True)
        tensor.add_label("disparity", disparity, align_dim=["B", "T"], mergeable=True)
        tensor.add_label("segmentation", segmentation, align_dim=["B", "T"], mergeable=False)

        # Add other tensor property
        tensor.add_property("normalization", normalization)
        tensor.add_property("_resnet_mean_std", ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
        tensor.add_property("mean_std", mean_std)

        return tensor

    def __init__(self, x, *args, **kwargs):
        super().__init__(x)

    def save(self, tgt_path: str):
        """Save the frame into a given location

        Parameters
        ----------
        tgt_path: str
            target path to save the frame. Example: /path/to/image.jpeg

        Notes
        -----
        The attached labels will not be saved.
        """
        torchvision.utils.save_image(self.cpu().norm01().as_tensor(), tgt_path)

    def append_boxes2d(self, boxes: BoundingBoxes2D, name: str = None):
        """Attach a set of boxes to the frame.

        Parameters
        ----------
        boxes: BoundingBoxes2D
            Boxes to attached to the Frame
        name: str
            If none, the boxes will be attached without name (if possible). Otherwise if no other unnamed
            boxes are attached to the frame, the boxes will be added to the set of boxes.
        """
        self._append_label("boxes2d", boxes, name)

    def append_boxes3d(self, boxes_3d: BoundingBoxes3D, name: str = None):
        """Attach boxes 3d to this image

        Parameters
        ----------
        boxes: BoundingBoxes3D
            Boxes to attached to the Frame
        name: str
            If none, the boxes will be attached without name (if possible). Otherwise if no other unnamed
            boxes are attached to the frame, the boxes will be added to the set of boxes.
        """
        self._append_label("boxes3d", boxes_3d, name)

    def append_mask(self, mask: Mask, name: str = None):
        """Attach a mask to the frame.

        Parameters
        ----------
        mask: aloscene.Mask
            Mask to attached to the Frame
        name: str
            If none, the mask will be attached without name (if possible). Otherwise if no other unnamed
            mask are attached to the frame, the mask will be added to the set of mask.
        """
        self._append_label("mask", mask, name)

    def append_flow(self, flow, name=None):
        """Attach a flow to the frame.

        Parameters
        ----------
        flow: aloscene.Flow
            Flow to attach to the Frame
        name: str
            If none, the flow will be attached without name (if possible). Otherwise if no other unnamed
            flow are attached to the frame, the flow will be added to the set of flow.
        """
        self._append_label("flow", flow, name)

    def append_disparity(self, disparity, name=None):
        """Attach a disparity map to the frame.

        Parameters
        ----------
        disparity: aloscene.Disparity
            Disparity to attach to the Frame
        name: str
            If none, the disparity will be attached without name (if possible). Otherwise if no other unnamed
            disparity are attached to the frame, the disparity will be added to the set of flow.
        """
        self._append_label("disparity", disparity, name)

    def append_segmentation(self, segmentation: Mask, name: str = None):
        """Attach a mask to the frame.

        Parameters
        ----------
        segmentation: aloscene.Mask
            Couple of mask to attached to the Frame
        name: str
            If none, the mask will be attached without name (if possible). Otherwise if no other unnamed
            mask are attached to the frame, the mask will be added to the set of mask.
        """
        self._append_label("segmentation", segmentation, name)

    @staticmethod
    def _get_mean_std_tensor(shape, names, mean_std: tuple, device="cpu"):
        """Utils method to a get the mean and the std
        reshaped properly to operate on a non-singleton dimension.
        """
        n_shape = [1] * len(shape)
        n_shape[names.index("C")] = 3
        mean_tensor = torch.tensor(mean_std[0], device=device).view(tuple(n_shape))
        std_tensor = torch.tensor(mean_std[1], device=device).view(tuple(n_shape))
        return mean_tensor, std_tensor

    def norm01(self) -> Frame:
        """Normnalize the tensor from the current tensor
        normalization to values between 0 and 1.
        """
        tensor = self

        if tensor.normalization == "01":
            return tensor.clone()
        elif tensor.normalization == "255":
            tensor = tensor.div(255)
        elif tensor.normalization == "minmax_sym":
            tensor = (tensor + 1.0) / 2.0
        elif tensor.mean_std is not None:
            mean_tensor, std_tensor = self._get_mean_std_tensor(
                tensor.shape, tensor.names, tensor._resnet_mean_std, device=tensor.device
            )
            tensor = tensor * std_tensor
            tensor = tensor + mean_tensor
        else:
            raise Exception(f"Can't convert from {tensor.normalization} to norm01")

        tensor.mean_std = None
        tensor.normalization = "01"

        return tensor

    def norm_as(self, target_frame) -> Frame:
        """Normalize the tensor as the given `target_frame`."""
        if target_frame.normalization == "01":
            return self.norm01
        elif target_frame.normalization == "255":
            return self.norm255()
        elif target_frame.normalization == "minmax_sym":
            return self.norm_minmax_sym()
        elif target_frame.mean_std is not None:
            return self.mean_std_norm(
                mean=target_frame.mean_std[0], std=target_frame.mean_std[1], name=target_frame.normalization
            )
        else:
            raise Exception(
                "Can't convert the tensor normalization to the"
                + f"target_frame normalization: {target_frame.normalization}"
            )

    def norm255(self) -> Frame:
        """Normnalize the tensor from the current tensor
        normalization to values between 0 and 255
        """
        tensor = self

        if tensor.normalization == "01":
            tensor = tensor.mul(255)
        elif tensor.normalization == "255":
            return tensor.clone()
        elif tensor.normalization == "minmax_sym":
            tensor = (tensor + 1.0) * 255.0 / 2.0
        elif tensor.mean_std is not None:
            mean_tensor, std_tensor = self._get_mean_std_tensor(
                tensor.shape, tensor.names, tensor._resnet_mean_std, device=tensor.device
            )
            tensor = tensor * std_tensor
            tensor = tensor + mean_tensor
            tensor = tensor.mul(255)
        else:
            raise Exception(f"Can't convert from {tensor.normalization} to norm255")

        tensor.mean_std = None
        tensor.normalization = "255"

        return tensor

    def norm_minmax_sym(self):
        """
        Normalize the tensor to values between -1 and 1
        """
        tensor = self
        if tensor.normalization == "minmax_sym":
            return tensor.clone()
        elif tensor.normalization == "01":
            tensor = 2 * tensor - 1.0
        elif tensor.normalization == "255":
            tensor = 2 * (tensor / 255.0) - 1.0
        else:
            raise Exception(f"Can't convert from {tensor.normalization} to norm255")
        tensor.mean_std = None
        tensor.normalization = "minmax_sym"
        return tensor

    def mean_std_norm(self, mean, std, name) -> Frame:
        """Normnalize the tensor from the current tensor
        normalization to the expected resnet normalization (x - mean) / std
        with x normalized with value between 0 and 1.
        """
        tensor = self
        mean_tensor, std_tensor = self._get_mean_std_tensor(
            tensor.shape, tensor.names, tensor._resnet_mean_std, device=tensor.device
        )
        if tensor.normalization == "01":
            tensor = tensor - mean_tensor
            tensor = tensor / std_tensor
        elif tensor.normalization == "255":
            tensor = tensor.div(255)
            tensor = tensor - mean_tensor
            tensor = tensor / std_tensor
        elif tensor.mean_std is not None and tensor.mean_std[0] == mean and tensor.mean_std[1] == std:
            return tensor.clone()
        elif tensor.mean_std is not None:
            tensor = tensor.norm01()
            tensor = tensor - mean_tensor
            tensor = tensor / std_tensor
        else:
            raise Exception(f"Can't convert from {tensor.normalization} to the given mean/std")

        tensor.normalization = name
        tensor.mean_std = (mean, std)

        return tensor

    def norm_resnet(self) -> Frame:
        return self.mean_std_norm(mean=self._resnet_mean_std[0], std=self._resnet_mean_std[1], name="resnet")

    def __get_view__(self, title=None):
        """Create a view of the frame"""
        assert self.names[0] != "T" and self.names[1] != "B"
        frame = self.norm01().cpu().rename(None).permute([1, 2, 0]).detach().contiguous().numpy()
        view = View(frame, title=title)
        return view

    def _pad(self, offset_y: tuple, offset_x: tuple, value=0, **kwargs):
        """Pad the based on the given offset

        Parameters
        ----------
        offset_y: tuple
            (percentage top_offset, percentage bottom_offset) Percentage based on the previous size
        offset_x: tuple
            (percentage left_offset, percentage right_offset) Percentage based on the previous size

        Returns
        -------
        padded
        """
        pad_values = {"01": 0, "255": 0, "minmax_sym": -1}
        if self.normalization in pad_values:
            pad_value = pad_values[self.normalization]
            return super()._pad(offset_y, offset_x, value=pad_value, **kwargs)
        elif self.mean_std is not None:
            # Set the new height and weight of the frame
            pad_top, pad_bottom, pad_left, pad_right = (
                round(offset_y[0] * self.H),
                round(offset_y[1] * self.H),
                round(offset_x[0] * self.W),
                round(offset_x[1] * self.W),
            )
            n_H = self.H + pad_top + pad_bottom
            n_W = self.W + pad_left + pad_right
            # Define the new shape of the padded frame
            n_tensor_shape = list(self.shape)
            n_tensor_shape[self.names.index("H")] = n_H
            n_tensor_shape[self.names.index("W")] = n_W
            # Create the new frame buffer
            n_tensor = torch.zeros(*tuple(n_tensor_shape), dtype=self.dtype, device=self.device)

            # Set the default value based on the frame normalization
            mean_tensor, std_tensor = self._get_mean_std_tensor(
                tuple(n_tensor_shape), self.names, self.mean_std, device=self.device
            )
            n_tensor = n_tensor - mean_tensor
            n_tensor = n_tensor / std_tensor

            # Copy the frame into the buffer
            n_slice = self.get_slices({"H": slice(pad_top, n_H - pad_bottom), "W": slice(pad_left, n_W - pad_right)})
            n_tensor[n_slice].copy_(self)

            # Create a new frame with the same parameters and set back a copy of the previous labels
            n_tensor = type(self)(n_tensor, normalization=self.normalization, mean_std=self.mean_std, names=self.names)
            n_tensor.set_labels(self.clone().get_labels())
            return n_tensor
        else:
            raise Exception("This normalziation {} is not handle by the frame _pad method".format(self.normalization))

    def _spatial_shift(self, shift_y: float, shift_x: float, **kwargs):
        """
        Spatially shift the Frame.

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

        x_shift = int(shift_x * self.W)
        y_shift = int(shift_y * self.H)

        frame_data = n_frame.as_tensor()

        permute_idx = list(range(0, len(self.shape)))
        last_current_idx = permute_idx[-1]
        permute_idx[-1] = permute_idx[self.names.index("C")]
        permute_idx[self.names.index("C")] = last_current_idx

        n_frame_mean = frame_data.permute(permute_idx)
        n_frame_mean = n_frame_mean.flatten(end_dim=-2)
        n_frame_mean = torch.mean(n_frame_mean, dim=0)
        n_shape = [1] * len(self.shape)
        n_shape[self.names.index("C")] = 3
        n_frame_mean = n_frame_mean.view(tuple(n_shape))

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
