from logging import warning
from shutil import ExecError
import matplotlib
from matplotlib.pyplot import sca
import torch
import warnings
import aloscene
from aloscene import Mask
from aloscene.renderer import View
from aloscene.utils.depth_utils import coords2rtheta
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
    is_bsolute: bool
        Either depth values refer to real values or shifted and scaled ones.
    scale: float
        Scale used to to shift depth. Pass this argument only if is_bsolute is set to True
    shift: float
        Intercept used to shift depth. Pass this argument only if is_bsolute is set to True
    """

    @staticmethod
    def __new__(
            cls,
            x,
            occlusion: Mask = None,
            is_absolute=True,
            is_planar=True,
            scale=None,
            shift=None,
            *args,
            names=("C", "H", "W"),
            **kwargs):
        if not is_absolute and (shift or scale):
            raise AttributeError('depth not in inverse state, can not pass scale or shift')
        if isinstance(x, str):
            x = load_depth(x)
            names = ("C", "H", "W")
        tensor = super().__new__(cls, x, *args, names=names, **kwargs)
        tensor.add_child("occlusion", occlusion, align_dim=["B", "T"], mergeable=True)
        tensor.add_property('scale', scale)
        tensor.add_property('shift', shift)
        tensor.add_property('is_absolute', is_absolute)
        tensor.add_property('is_planar', is_planar)
        return tensor

    def __init__(self, x, *args, **kwargs):
        super().__init__(x)

    def encode_inverse(self, prior_clamp_min=None, prior_clamp_max=None, post_clamp_min=None, post_clamp_max=None):
        """Undo encode_absolute tansformation

        Parameters
        ----------
        prior_clamp_min: float | None
            Clamp min depth before to convert to idepth
        prior_clamp_max: float | None
            Clamp max depth before to convert to idepth
        post_clamp_min: float | None
            Clamp min output idepth
        post_clamp_max: float | None
            Clamp max output idepth

        Exemples
        -------
        >>> not_absolute_depth = Depth(torch.ones((1, 1, 1)), is_absolute=False)
        >>> absolute_depth = not_absolute_depth.encode_absolute()
        >>> undo_depth = absolute_depth.encode_inverse()
        >>> (undo_depth == not_absolute_depth).item()
        >>> True
        """
        if not depth.is_absolute:
            print('No need to inverse depth, already inversed')
            return self
        depth = self
        shift = depth.shift if depth.shift is not None else 0
        scale = depth.scale if depth.scale is not None else 1

        if prior_clamp_min is not None or prior_clamp_max is not None:
            depth = torch.clamp(depth, min=prior_clamp_min, max=prior_clamp_max)

        depth = 1 / depth
        depth = (depth - shift) / scale

        if post_clamp_min is not None or post_clamp_max is not None:
            depth = torch.clamp(depth, min=post_clamp_min, max=post_clamp_max)

        depth.scale = None
        depth.shift = None
        depth.is_absolute = False
        return depth

    def encode_absolute(self, scale=1, shift=0, prior_clamp_min=None, prior_clamp_max=None, post_clamp_min=None, post_clamp_max=None, keep_negative=False):
        """Transforms inverted depth to absolute depth

        Parameters
        ----------
            scale: (: float)
                Multiplication factor. Default is 1.
            shift: (: float)
                Addition intercept. Default is 0.
            prior_clamp_min: float | None
                Clamp min idepth before to convert to depth
            prior_clamp_max: float | None
                Clamp max idepth before to convert to depth
            post_clamp_min: float | None
                Clamp min output idepth
            post_clamp_max: float | None
                Clamp max output idepth
            keep_negative: bool | False
                Keep negative plannar depth (points behind camera, useful for wide angle lens with FoV bigger
                than 180 degree)

        Exemples
        --------
        >>> not_absolute_depth = Depth(torch.ones((1, 20, 20)), is_absolute=False)
        >>> absolute_depth = not_absolute_depth.encode_absolute()
        >>> absolute_depth.is_absolute, not_absolute_depth.is_absolute
        >>> True, False
        """
        if depth.is_absolute:
            print('Depth already in absolute value.')
            return self
        depth, names = self.rename(None), self.names

        depth = depth * scale + shift

        if prior_clamp_min is not None or prior_clamp_max is not None:
            depth = torch.clamp(depth, min=prior_clamp_min, max=prior_clamp_max)

        if keep_negative and self.is_planar:
            depth[torch.unsqueeze((depth < 1e-8) & (depth >= 0), dim=0)] = 1e-8
            depth[torch.unsqueeze((depth >= -1e-8) & (depth < 0), dim=0)] = -1e-8
        else:
            depth[torch.unsqueeze(depth < 1e-8, dim=0)] = 1e-8

        depth.scale = scale
        depth.shift = shift
        depth.is_absolute = True

        n_depth = (1 / depth).rename(*names)

        if post_clamp_min is not None or post_clamp_max is not None:
            n_depth = torch.clamp(n_depth, min=post_clamp_min, max=post_clamp_max)

        return n_depth

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

    def as_euclidean(self, camera_intrinsic: aloscene.CameraIntrinsic = None, projection="pinhole", distortion=1.0):
        """Create a new Depth augmented tensor whose data is the euclidean depth (distance) from camera to world points.
        To use this method, we must know intrinsic matrix of camera, projection model and distortion coefficient
        (if exists).

        Parameters
        ----------
        camera_intrinsic: aloscene.CameraIntrinsic
        projection: str | pinhole
            At this moment, only 2 projections models are supported: pinhole (f*tan(theta)) and equidistant (f*theta:
            which is
            used for wide range camera).
        distortion: float | 1.0
            Distortion coefficient for equidistant model. Only linear distortion supported
            (sensor_angle=distortion*theta).

        Returns
        -------
        aloscene.Depth
        """
        planar = self
        if not planar.is_planar:
            print("This tensor is already a euclidian depth tensor so no transform is performed")
            return planar
        assert projection in ["pinhole", "equidistant"], "Only pinhole and equidistant projection are supported"

        camera_intrinsic = camera_intrinsic if camera_intrinsic is not None else self.cam_intrinsic
        if camera_intrinsic is None:
            err_msg = "The `camera_intrinsic` must be given either from the current depth tensor or from "
            err_msg += "the as_disp(camera_intrinsic=...) method."
            raise Exception(err_msg)

        _, theta = coords2rtheta(camera_intrinsic, planar.HW, distortion, projection)
        euclidean = planar / (torch.cos(theta) + 1e-8)
        euclidean.is_planar = False
        return euclidean

    def as_planar(self, camera_intrinsic: aloscene.CameraIntrinsic = None, projection="pinhole", distortion=1.0):
        """Create a new planar depth augmented tensor from the euclidean depth between camera to world points with
        corresponding depth. To use this method, we must know intrinsic matrix of camera, projection model and
        distortion coefficient (if exists).

        Parameters
        ----------
        camera_intrinsic: aloscene.CameraIntrinsic
        projection: str | pinhole
            At this moment, only 2 projections models are supported: pinhole (f*tan(theta)) and equidistant (f*theta:
            which is used for wide range camera).
        distortion: float | 1.0
            Distortion coefficient for equidistant model. Only linear distortion supported
            (sensor_angle=distortion*theta).

        Returns
        -------
        aloscene.Depth
        """
        euclidean = self
        if euclidean.is_planar:
            print("This tensor is already a planar depth tensor so no transform is done.")
            return euclidean
        assert projection in ["pinhole", "equidistant"], "Only pinhole and equidistant projection are supported"

        camera_intrinsic = camera_intrinsic if camera_intrinsic is not None else self.cam_intrinsic
        if camera_intrinsic is None:
            err_msg = "The `camera_intrinsic` must be given either from the current depth tensor or from "
            err_msg += "the as_disp(camera_intrinsic=...) method."
            raise Exception(err_msg)

        _, theta = coords2rtheta(camera_intrinsic, euclidean.HW, distortion, projection)
        planar = euclidean * torch.cos(theta)
        planar.is_planar = True
        return planar
