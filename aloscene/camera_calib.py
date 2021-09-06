from __future__ import annotations
from aloscene.tensors.augmented_tensor import AugmentedTensor


class CameraIntrinsic(AugmentedTensor):
    """
    Camera Intrinsic matrix of shape [..., 3, 4]
    The intrinsic matrix transforms 3D camera cooordinates to 2D homogeneous image coordinates.
    This perspective projection is modeled by the ideal pinhole camera

    The last two dimensions is:\n
        fx   s    x0   0\n
        0    fy   y0   0\n
        0    0     1   0

    - Focal length: fx, fy. For true pinhole camera, fx and fy have the same value.
    - Princial Point Offset: x0, y0
    - Axis skew: s, in most cases, this value should be 0

    For more information: https://ksimek.github.io/2013/08/13/intrinsic/
    """

    @staticmethod
    def __new__(cls, x, *args, **kwargs):
        tensor = super().__new__(cls, x, *args, **kwargs)
        return tensor

    def __init__(self, x, *args, **kwargs):
        assert x.shape[-2] == 3 and x.shape[-1] == 4
        super().__init__(x)

    def _hflip(self, *args, frame_size: tuple[int, int], **kwargs):
        """
        frame_size: (H, W)
        """
        cam_intrinsic = self.clone()
        cam_intrinsic[..., 0, 2] = frame_size[1] - cam_intrinsic[..., 0, 2]
        return cam_intrinsic

    def _resize(self, size, **kwargs):
        cam_intrinsic = self.clone()
        resize_ratio_w = size[0]
        resize_ratio_h = size[1]
        assert (
            abs(resize_ratio_h - resize_ratio_w) < 1e-2
        ), f"Should resize with the same aspect ratio, got ratio: {size}"
        cam_intrinsic[..., 0, 0] *= resize_ratio_h  # fx
        cam_intrinsic[..., 1, 1] *= resize_ratio_h  # fy
        cam_intrinsic[..., 0, 2] *= resize_ratio_h  # x0
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

    def _pad(self, offset_y, offset_x, frame_size: tuple[int, int], **kwargs):
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
