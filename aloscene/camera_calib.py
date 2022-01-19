# from __future__ import annotations
from typing import Tuple
from aloscene.tensors.augmented_tensor import AugmentedTensor
import aloscene


import torch
import numpy as np


class CameraIntrinsic(AugmentedTensor):
    """
    Euclidian Camera Intrinsic matrix of shape [..., 3, 4]
    The intrinsic matrix transforms 3D camera cooordinates to 2D homogeneous image coordinates.
    This perspective projection is modeled by the euclidian pinhole camera

    The last two dimensions is:\n
        fx   s    x0   0\n
        0    fy   y0   0\n
        0    0     1   0

    - Focal length: fx, fy. For true pinhole camera, fx and fy have the same value.
    - Princial Point Offset: x0, y0
    - Axis skew: s, in most cases, this value should be 0

    For more information: https://ksimek.github.io/2013/08/13/intrinsic/

    Parameters
    x: torch.Tensor | np.array | None
        Could be None. If None, x will be created based on `focal_length`, `plane_size` & the `principal_point`.
    focal_length: float | tuple | None
        Focal length (in px) of the current frame. If tuple, will be (fy, fx).
    plane_size: tuple | None
        The `plane size` (in px) is used to unproject or project points from 2D to 3D (or 3D to 2D). If the
        `plane_size` is not provided, it will be assume to be equal to the current
        tensor size. In some cases, this asumption might be Fase. If so, one must pass the plane_size using one tuple
        (height, width).
    principal_point: tuple | None
        Coordinates of the plane center in the following format : (x0, y0), (in px). If none,
        the principal points will be assume to be half the size of the plane (if possible). If the plane size is not
        given, the principal points will be zero.
    skew: float | None
        This is the shear time the camera constant.

    """

    @staticmethod
    def __new__(
        cls,
        x=None,
        focal_length: float = None,
        plane_size: tuple = None,
        principal_point: tuple = None,
        skew: tuple = None,
        *args,
        names=(None, None),
        **kwargs,
    ):
        if x is None:
            x = torch.zeros((4, 4))
            focal_length = (focal_length, focal_length) if not isinstance(focal_length, tuple) else focal_length
            x[0][0] = focal_length[1] if focal_length[1] is not None else np.inf
            x[1][1] = focal_length[0] if focal_length[1] is not None else np.inf
            x[0][1] = skew if skew is not None else 0

            if principal_point is None and plane_size is not None:
                if plane_size[0] % 2 != 0 or plane_size[1] % 2 != 0:
                    err = "The principal points (center of the plane) can't be infer automaticly."
                    err += " The latter must be given when creating the CameraIntrinsic tensor."
                    raise Exception(err)
                principal_point = (plane_size[0] / 2, plane_size[1] / 2)
            elif principal_point is None:
                principal_point = (0, 0)
            x[0][2] = principal_point[1]
            x[1][2] = principal_point[0]
            x[2][2] = 1
            x[3][3] = 1

        tensor = super().__new__(cls, x, *args, names=names, **kwargs)
        return tensor

    def __init__(self, x=None, *args, **kwargs):
        if x is not None:
            assert x.shape[-2] == 3 and x.shape[-1] == 4
        super().__init__(x)

    @property
    def focal_length(self) -> torch.Tensor:
        """Focal length with the last dimensions being fx and fy"""
        return self.as_tensor()[..., (0, 1), (0, 1)]

    @property
    def principal_points(self) -> torch.Tensor:
        """Principal point with the last dimensions being x0 and y0"""
        return self.as_tensor()[..., (0, 1), (2, 2)]

    @property
    def skew(self) -> torch.Tensor:
        """Skew (shear time camera constant)"""
        return self.as_tensor()[..., 0, 1]

    def _hflip(self, *args, frame_size: Tuple[int, int], **kwargs):
        """
        frame_size: (H, W)
        """
        assert abs(self.skew) < 1e-3
        cam_intrinsic = self.clone()
        cam_intrinsic[..., 0, 2] = frame_size[1] - cam_intrinsic[..., 0, 2]
        return cam_intrinsic

    def _vflip(self, *args, frame_size: Tuple[int, int], **kwargs):
        """
        frame_size: (H, W)
        """
        assert abs(self.skew) < 1e-3
        cam_intrinsic = self.clone()
        cam_intrinsic[..., 1, 2] = frame_size[0] - cam_intrinsic[..., 1, 2]
        return cam_intrinsic

    def _resize(self, size, **kwargs):
        cam_intrinsic = self.clone()
        resize_ratio_w = size[0]
        resize_ratio_h = size[1]
        cam_intrinsic[..., 0, 0] *= resize_ratio_w  # fx
        cam_intrinsic[..., 1, 1] *= resize_ratio_h  # fy
        cam_intrinsic[..., 0, 2] *= resize_ratio_w  # x0
        cam_intrinsic[..., 1, 2] *= resize_ratio_h  # y0
        return cam_intrinsic

    def _crop(self, H_crop, W_crop, frame_size, **kwargs):
        """Crop the SpatialAugmentedTensor

        Parameters
        ----------
        H_crop: tuple
            (start, end) between 0 and 1
        W_crop: tuple
            (start, end) between 0 and 1
        frame_size: tuple (H, W)
        """
        hmin = H_crop[0] * frame_size[0]
        wmin = W_crop[0] * frame_size[1]
        cam_intrinsic = self.clone()
        cam_intrinsic[..., 0, 2] -= wmin
        cam_intrinsic[..., 1, 2] -= hmin
        return cam_intrinsic

    def _pad(self, offset_y, offset_x, frame_size: Tuple[int, int], **kwargs):
        """Pad the set of boxes based on the given offset

        Parameters
        ----------
        offset_y: tuple
            (percentage top_offset, percentage bottom_offset) Percentage based on the previous size
        offset_x: tuple
            (percentage left_offset, percentage right_offset) Percentage based on the previous size
        frame_size: tuple (H, W)
        Returns
        -------
        padded_cam_intrinsic: CameraIntrinsic
        """
        pad_top = offset_y[0] * frame_size[0]
        pad_left = offset_x[0] * frame_size[1]
        cam_intrinsic = self.clone()
        cam_intrinsic[..., 0, 2] += pad_top
        cam_intrinsic[..., 1, 2] += pad_left
        return cam_intrinsic

    def get_view(self, *args, **kwargs):
        pass


class CameraExtrinsic(AugmentedTensor):
    """
    Camera Extrinsic of shape [..., 4, 4].
    This matrix transforms real world coordinates to camera coordinates by translation and rotation.

    The last two dimension forms a transformation matrix (translation vector and rotation matrix):\n
        R11 R12 R13  t1\n
        R21 R22 R23  t2\n
        R31 R32 R33  t3\n
        0   0   0    1
    """

    @staticmethod
    def __new__(cls, x, *args, **kwargs):
        tensor = super().__new__(cls, x, *args, **kwargs)
        return tensor

    def __init__(self, x, *args, **kwargs):
        assert x.shape[-2] == 4 and x.shape[-1] == 4
        super().__init__(x)

    def _hflip(self, *args, **kwargs):
        return self.clone()

    def _resize(self, *args, **kwargs):
        # Resize image does not change cam extrinsic
        return self.clone()

    def _crop(self, *args, **kwargs):
        # Cropping image does not change cam extrinsic
        return self.clone()

    def _pad(self, *args, **kwargs):
        # Padding image does not change cam extrinsic
        return self.clone()

    def get_view(self, *args, **kwargs):
        pass
