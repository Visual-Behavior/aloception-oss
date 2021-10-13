import matplotlib
import torch

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
    def __new__(cls, x, occlusion: Mask = None, *args, names=("C", "H", "W"), **kwargs):
        if isinstance(x, str):
            x = load_depth(x)
            names = ("C", "H", "W")
        tensor = super().__new__(cls, x, *args, names=names, **kwargs)
        tensor.add_label("occlusion", occlusion, align_dim=["B", "T"], mergeable=True)

        return tensor

    def __init__(self, x, *args, **kwargs):
        super().__init__(x)

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
        self._append_label("occlusion", occlusion, name)

    def __get_view__(self, cmap="nipy_spectral", min_depth=0, max_depth=200):
        assert all(dim not in self.names for dim in ["B", "T"]), "Depth should not have batch or time dimension"
        cmap = matplotlib.cm.get_cmap(cmap)
        depth = self.rename(None).permute([1, 2, 0]).detach().cpu().contiguous().numpy()

        depth = max_depth - np.clip(depth, min_depth, max_depth)
        depth = matplotlib.colors.Normalize(vmax=max_depth)(depth)
        depth_color = cmap(depth)[:, :, 0, :3]
        return View(depth_color)

    def as_points3d(self, plane_size: tuple = None, principal_point=None, focal_length: float = None):
        """Compute the 3D coordinates of points 2D points based on their respective depth.

        Parameters
        ----------
        x_points : {np.ndarray}
            Numpy array with the x coordinates of 2d points
        y_points : {np.ndarray}
            Numpy array with the y coordinates of 2d points
        z_points : {np.ndarray}
            Numpy array with the z depth of 2d points

        Returns
        -------
        points_3d : {np.ndarray}
            (n, 3) with the 3d coordinates [x, y, z] of each provided 2d point.
        """
        focal_length = focal_length if focal_length is not None else self.focal_length
        principal_point = principal_point if principal_point is not None else self.principal_point
        plane_size = plane_size if plane_size is not None else self.plane_size

        y_points, x_points = torch.meshgrid(torch.arange(self.H), torch.arange(self.W))
        if plane_size is not None and self.HW != plane_size:
            y_points = y_points * plane_size[0] / self.H
            x_points = x_points * plane_size[1] / self.W

        z_points = self.as_tensor().view((-1, self.H * self.W))

        if focal_length is None or plane_size is None:
            err = "The focal_length or plane_size can't be infer automaticly."
            err += " The latter must be given when creating the Depth tensor or when calling "
            err += "as_point3d(focal_length=...,  plane_size=...)"
            raise Exception(err)

        if principal_point is None:  # Try to automaticly set the principal_point
            if plane_size[0] % 2 != 0 or plane_size[1] % 2 != 0:
                err = "The principal points (center of the plane) can't be infer automaticly."
                err += " The latter must be given when creating the Depth tensor or when calling "
                err += "as_point3d(principal_point=(c_y, c_x))"
                raise Exception(err)
            principal_point = (self.plane_size[0] / 2, self.plane_size[1] / 2)

        target_shape = tuple([self.shape[self.names.index(n)] for n in self.names if n not in ("C", "H", "W")] + [-1])
        # Broadcasted shape
        broad_shape = tuple([1 for n in self.names if n not in ("C", "H", "W")] + [-1])
        target_names = tuple([n for n in self.names if n not in ("C", "H", "W")] + ["N", None])

        y_points = y_points.reshape(broad_shape)
        x_points = x_points.reshape(broad_shape)
        z_points = self.as_tensor().reshape(target_shape)

        points_3d_shape = tuple(list(target_shape)[:-1] + [self.H * self.W] + [3])
        points_3d = torch.zeros(points_3d_shape, device=self.device)

        points_3d[..., 0] = x_points - principal_point[1]
        points_3d[..., 1] = y_points - principal_point[0]
        points_3d[..., 2] = z_points
        points_3d[..., :2] = points_3d[..., :2] * points_3d[..., 2:] / focal_length

        return aloscene.Points3D(
            points_3d,
            names=target_names,
        )

    def as_disp(
        self,
        camera_side: str = None,
        baseline: float = None,
        focal_length: float = None,
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
        focal_length: float | None
            The `focal_length` must be known to convert this depth tensor into a disparity tensor. The `focal_length`
            must be given either from this current depth tensor or from this parameter.

        Returns
        -------
        aloscene.Disparity
        """
        baseline = baseline if baseline is not None else self.baseline
        focal_length = focal_length if focal_length is not None else self.focal_length
        camera_side = camera_side if camera_side is not None else self.camera_side
        if baseline is None:
            raise Exception(
                "Can't convert to disparity. The `baseline` must be given either from the current depth tensor or from the as_disp(baseline=...) method."
            )
        if focal_length is None:
            raise Exception(
                "The `focal_length` must be given either from the current depth tensor or from this the as_disp(focal_length=...) method."
            )

        depth = aloscene.Disparity(
            baseline * focal_length / self.clone().as_tensor(),
            baseline=baseline,
            focal_length=focal_length,
            camera_side=camera_side,
            disp_format="unsigned",
            names=self.names,
        )
        return depth
