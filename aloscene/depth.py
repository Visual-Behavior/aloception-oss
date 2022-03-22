from logging import warning
import matplotlib
from matplotlib.pyplot import sca
import torch
import warnings
import aloscene
from aloscene import Mask
from aloscene.renderer import View
import numpy as np

from aloscene.io.depth import load_depth


class Depth(aloscene.tensors.SpatialAugmentedTensor):
    """
    Depth map

    Parameters
    ----------
    x: tensor or str
        loaded Depth tensor or path to the Depth file from which Depth will be loaded.
    occlusion : aloscene.Mask
        Occlusion mask for this Depth map. Default value : None.
    """

    @staticmethod
    def __new__(
            cls, 
            x, 
            occlusion: Mask = None, 
            is_inverse=False, 
            scale=None, 
            shift=None, 
            *args, 
            names=("C", "H", "W"), 
            **kwargs):
        if is_inverse and not (shift and scale):
            raise AttributeError('inversed depth requires shift and scale arguments')
        if isinstance(x, str):
            x = load_depth(x)
            names = ("C", "H", "W")
        tensor = super().__new__(cls, x, *args, names=names, **kwargs)
        tensor.add_child("occlusion", occlusion, align_dim=["B", "T"], mergeable=True)
        tensor.add_property('_scale', scale)
        tensor.add_property('_shift', shift)
        tensor.add_property('_is_inverse', is_inverse)
        return tensor

    def __init__(self, x, *args, **kwargs):
        super().__init__(x)

    def encode_inverse(
            self, 
            scale: float = 1, 
            shift: float = 0):
        """Shift and scales the depth values"""
        depth = self.detach().cpu()
        if depth._is_inverse:
            depth = self.encode_absolute()
        depth.add_property('_scale', scale)
        depth.add_property('_shift', shift)
        depth.add_property('_is_inverse', True)
        depth = depth * scale + shift
        return 1 / depth
    
    def encode_absolute(self):
        """Undo inverse encoding"""
        depth = self.detach().cpu()
        if not depth._is_inverse:
            warnings.warn('depth already encoded in absolute mode')
            return depth
        depth = 1 / depth
        depth = depth / self.scale - self.shift
        depth.add_property('_scale', None)
        depth.add_property('_shift', None)
        depth.add_property('_is_inverse', False)
        return depth

    def append_occlusion(self, occlusion: Mask, name: str = None):
        """Attach an occlusion mask to the depth tensor.

        Parameters
        ----------
        occlusion: Mask
            Occlusion mask to attach to the Frame
        name: str
            If none, the occlusion mask will be attached without name (if possible). Otherwise if no other unnamed
            occlusion mask are attached to the frame, the mask will be added to the set of mask.
        """
        self._append_child("occlusion", occlusion, name)

    def __get_view__(self, cmap="nipy_spectral", min_depth=0, max_depth=200, title=None, reverse=True):
        assert all(dim not in self.names for dim in ["B", "T"]), "Depth should not have batch or time dimension"
        cmap = matplotlib.cm.get_cmap(cmap)
        depth = self.rename(None).permute([1, 2, 0]).detach().cpu().contiguous().numpy()
        depth = matplotlib.colors.Normalize(vmin=min_depth, vmax=max_depth, clip=True)(depth)
        if reverse:
            depth = 1 - depth
        depth_color = cmap(depth)[:, :, 0, :3]
        return View(depth_color, title=title)

    def as_points3d(self, camera_intrinsic: aloscene.CameraIntrinsic = None):
        """Compute the 3D coordinates of points 2D points based on their respective depth.

        Parameters
        ----------
        camera_intrinsic: CameraIntrinsic to use to unproject the points to 3D. If not, will try to use
        the instance `cam_intrinsic` if set.

        Returns
        -------
        points_3d : {np.ndarray}
            (n, 3) with the 3d coordinates [x, y, z] of each provided 2d point.
        """
        intrinsic = camera_intrinsic if camera_intrinsic is not None else self.cam_intrinsic

        y_points, x_points = torch.meshgrid(
            torch.arange(self.H, device=self.device), torch.arange(self.W, device=self.device)
        )

        z_points = self.as_tensor().view((-1, self.H * self.W))

        if intrinsic is None:
            err_msg = "The `camera_intrinsic` must be given either from the current depth tensor or from "
            err_msg += "the as_points3d(camera_intrinsic=...) method."
            raise Exception(err_msg)

        target_shape = tuple([self.shape[self.names.index(n)] for n in self.names if n not in ("C", "H", "W")] + [-1])
        target_names = tuple([n for n in self.names if n not in ("C", "H", "W")] + ["N", None])

        y_points = y_points.reshape((-1,))
        x_points = x_points.reshape((-1,))
        z_points = self.as_tensor().reshape(target_shape)

        # Append batch & temporal dimensions
        for _ in range(len(target_shape[:-1])):
            y_points = y_points.unsqueeze(0)
            x_points = x_points.unsqueeze(0)

        points_3d_shape = tuple(list(target_shape)[:-1] + [self.H * self.W] + [3])
        points_3d = torch.zeros(points_3d_shape, device=self.device)

        # Principal points & Focal length. Flatten the temporal & Batch dimension (If any)
        principal_points = intrinsic.principal_points
        focal_length = intrinsic.focal_length
        if len(intrinsic.shape) > 2:
            principal_points = principal_points.flatten(0, -2)
            focal_length = focal_length.flatten(0, -2)

        points_3d[..., 0] = x_points - principal_points[..., 0:1]
        points_3d[..., 1] = y_points - principal_points[..., 1:]
        points_3d[..., 2] = z_points
        points_3d[..., :2] = points_3d[..., :2] * points_3d[..., 2:] / focal_length.unsqueeze(-2)

        return aloscene.Points3D(points_3d, names=target_names, device=self.device)

    def as_disp(
        self, camera_side: str = None, baseline: float = None, camera_intrinsic: aloscene.CameraIntrinsic = None
    ):
        """Create a disparity augmented tensor from the current Depth augmented tensor.
        To use this method, one must know the target `camera_side` ("left" or "right"). Also, if not set on the
        tensor, the focal_length & the baseline must be given.

        Parameters
        ----------
        camera_side : str | None
            If created from a stereo camera, this information can optionally be used to convert
            this depth tensor into a disparity tensor. The `camera_side` is necessary to switch from unsigned to signed
            format once using a disparity tensor.
        baseline: float | None
            The `baseline` must be known to convert this depth tensor into disparity
            tensor. The `baseline` must be given either from from the current depth tensor or from this parameter.
        camera_intrinsic: aloscene.CameraIntrinsic
            CameraIntrinsic use to transform the depth map into disparity map using the intrinsic focal
            length.

        Returns
        -------
        aloscene.Disparity
        """
        baseline = baseline if baseline is not None else self.baseline
        camera_side = camera_side if camera_side is not None else self.camera_side
        intrinsic = camera_intrinsic if camera_intrinsic is not None else self.cam_intrinsic

        if intrinsic is None:
            err_msg = "The `camera_intrinsic` must be given either from the current depth tensor or from "
            err_msg += "the as_disp(camera_intrinsic=...) method."
            raise Exception(err_msg)
        if baseline is None:
            err_msg = "Can't convert to disparity. The `baseline` must be given either from the current depth tensor "
            err_msg += "or from the as_disp(baseline=...) method."
            raise Exception(err_msg)

        # unsqueeze on the Spatial H,W Dimension. On the dimension before "C", the focal_length is already supposed
        # to be aligned properly.
        focal_length = intrinsic.focal_length[..., 0:1].unsqueeze(-1).unsqueeze(-1)

        depth = aloscene.Disparity(
            baseline * focal_length / self.clone().as_tensor(),
            baseline=baseline,
            camera_side=camera_side,
            disp_format="unsigned",
            cam_intrinsic=intrinsic,
            cam_extrinsic=self.cam_extrinsic,
            names=self.names,
        )
        return depth
