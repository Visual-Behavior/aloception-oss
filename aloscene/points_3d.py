from __future__ import annotations

from torchvision.io.image import read_image
import torch
from torch import Tensor
from torch._C import device
import torchvision

from typing import *
import numpy as np
import cv2

import aloscene
from aloscene.renderer import View
from aloscene.labels import Labels
import torchvision
from torchvision.ops.boxes import nms
from aloscene.renderer import View, put_adapative_cv2_text, adapt_text_size_to_frame
import itertools


class Points3D(aloscene.tensors.AugmentedTensor):
    """Points3D Augmented Tensor. Used to represents 3D points (points cloud) in space encoded as xyz. The data must be
    at least 2 dimensional (N, None) where N is the number of points.

    If your're data is more than 2 dimensional you might need to set the `names` to ("B", "N", None) for batch
    dimension or ("T", "N", None) for the temporal dimension, or even ("B", "T", "N", None) for batch and temporal
    dimension. The latter works as far as you expected to get the same number of points for each Batch or time
    step of your sequence. Otherwise, you might want to enclose your Points3D into a list:

    >>> [Points3D(...), Points3D(...), Points3D(...)].

    Finally,  batch & temporal dimension could also be stored like this:

    >>> [[Points3D(...), Points3D(...)], [Points3D(...), Points3D(...)]]


    Parameters
    ----------
    x: list | torch.Tensor | np.array
        Points3D data. See explanation above for details.

    names: tuple
        Names of the dimensions : ("N", None) by default. See explanation above for more details.

    Notes
    -----
    Note on dimension:

    - C refers to the channel dimension
    - N refers to a dimension with a dynamic number of element.
    - H refers to the height of a `SpatialAugmentedTensor`
    - W refers to the width of a `SpatialAugmentedTensor`
    - B refers to the batch dimension
    - T refers to the temporal dimension
    """

    @staticmethod
    def __new__(
        cls,
        x: Union[list, np.array, torch.Tensor],
        names=("N", None),
        *args,
        **kwargs,
    ):

        tensor = super().__new__(cls, x, *args, names=names, **kwargs)
        return tensor

    def __init__(self, x, *args, **kwargs):
        super().__init__(x)

    def as_depth(self, base_depth: aloscene.Depth, camera_intrinsic: aloscene.CameraIntrinsic):
        """Project back the points onto the image plane. The results image will
        be returns as an aloscene.Depth map.

        Parameters
        ----------
        base_depth: aloscene.Depth
            This depth map will be used to project the 3D points.
            One can choose to set the default value of the `base_depth` to zero, infinity, nan, or to any relevant value.
        camera_intrinsic: aloscene.CameraIntrinsic
            CameraIntrinsic to use to unproject the points to 3D. If not, will try to use the instance
            `cam_intrinsic` if set.
        """
        principal_points = camera_intrinsic.principal_points.unsqueeze(-2)
        focal_length = camera_intrinsic.focal_length.unsqueeze(-2)

        # Project
        # coordinates in image plane, (in pixels)
        projected_points = self.as_tensor()
        projected_points[..., :2] = projected_points[..., :2] / projected_points[..., 2:3] * focal_length

        # move frame origin from image center to top-left corner
        projected_points[..., 0] = projected_points[..., 0] + principal_points[..., 0]
        projected_points[..., 1] = projected_points[..., 1] + principal_points[..., 1]

        slice_dim = {}
        N = self.shape[self.names.index("N")]
        for d, dim_name in enumerate(self.names):
            if dim_name != "N" and dim_name is not None:
                slice_dim[dim_name] = torch.as_tensor(
                    list(itertools.chain.from_iterable([[i] * N for i in range(self.shape[d])])), device=self.device
                )

        flattened_projected_points = projected_points.flatten(0, len(projected_points.shape) - 2)
        flattened_projected_points = flattened_projected_points.to(torch.float32)
        slice_dim["H"] = torch.round(flattened_projected_points[:, 1])
        slice_dim["W"] = torch.round(flattened_projected_points[:, 0])

        valid_points = (
            (slice_dim["H"] >= 0)
            & (slice_dim["H"] < base_depth.H)
            & (slice_dim["W"] >= 0)
            & (slice_dim["W"] < base_depth.W)
        )
        slice_dim = {key: slice_dim[key][valid_points].to(torch.int64) for key in slice_dim}

        depth_slice = base_depth.get_slices(slice_dim)
        n_base_depth_names = base_depth.names
        n_base_depth = base_depth.as_tensor()

        if n_base_depth_names.index("C") == 0:
            n_base_depth[depth_slice] = flattened_projected_points[valid_points][..., 2:].T
        else:
            n_base_depth[depth_slice] = flattened_projected_points[valid_points][..., 2:]

        return aloscene.Depth(
            n_base_depth,
            cam_intrinsic=camera_intrinsic,
            names=n_base_depth_names,
        )

    def get_view(self, **kwargs):
        return None

    def _hflip(self, **kwargs):
        """Flip points horizontally"""
        raise Exception("Not handle yet")

    def _resize(self, size, **kwargs):
        raise Exception("Not handler yet")

    def _crop(self, H_crop: tuple, W_crop: tuple, **kwargs) -> Points3D:
        raise Exception("Not handler yet")

    def _pad(self, offset_y: tuple, offset_x: tuple, **kwargs) -> Points3D:
        raise Exception("Not handler yet")

    def _spatial_shift(self, shift_y: float, shift_x: float, **kwargs):
        raise Exception("Not handle yet")
