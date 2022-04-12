from aloscene.camera_calib import CameraExtrinsic
import torch


class Pose(CameraExtrinsic):
    """Pose Tensor. Usually use to store World2Frame coordinates

    Parameters
    ----------
    x: torch.Tensor
        Pose matrix
    """

    @staticmethod
    def __new__(cls, x, *args, names=(None, None), **kwargs):
        tensor = super().__new__(cls, x, *args, names=names, **kwargs)
        return tensor

    def __init__(self, x, *args, **kwargs):
        super().__init__(x)

    def _hflip(self, *args, **kwargs):
        return self.clone()

    def _vflip(self, *args, **kwargs):
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
