import numpy as np
import matplotlib
import torch

import aloscene
from aloscene import Mask
from aloscene.renderer import View

from aloscene.io.disparity import load_disp


class Disparity(aloscene.tensors.SpatialAugmentedTensor):
    """
    Disparity map

    Parameters
    ----------
    x: tensor or str
        loaded disparity tensor or path to the disparity file from which disparity will be loaded.
    occlusion : aloscene.Mask
        Occlusion mask for this disparity map. Default value : None.
    disp_format : {'signed'|'unsigned'}
        If unsigned, disparity is interpreted as a distance (positive value) in pixels.
        If signed, disparity is interpreted as a relative offset (values can be negative).
    png_negate: bool
        if true, the sign of disparity is reversed when loaded from file.
        this parameter should be explicitely set every time a .png file is used.
    """

    @staticmethod
    def __new__(
        cls,
        x,
        occlusion: Mask = None,
        disp_format="unsigned",
        png_negate: bool = None,
        *args,
        names=("C", "H", "W"),
        **kwargs,
    ):
        if isinstance(x, str):
            x = load_disp(x, png_negate)
            names = ("C", "H", "W")

        tensor = super().__new__(cls, x, *args, names=names, **kwargs)
        tensor.add_child("occlusion", occlusion, align_dim=["B", "T"], mergeable=True)
        tensor.add_property("disp_format", disp_format)

        if tensor.disp_format == "unsigned" and (tensor.as_tensor() < 0).any():
            raise ValueError("All disparity values should be positive for disp_format='unsigned'")
        if tensor.disp_format == "signed" and tensor.camera_side is None:
            raise ValueError("camera_side is needed for signed disparity")
        if tensor.disp_format == "signed":
            vmin, vmax = (None, 0) if tensor.camera_side == "left" else (0, None)
            tensor = torch.clamp(tensor, vmin, vmax)
        return tensor

    def __init__(self, x, *args, **kwargs):
        super().__init__(x)

    def append_occlusion(self, occlusion: Mask, name: str = None):
        """Attach an occlusion mask to the frame.

        Parameters
        ----------
        occlusion: Mask
            Occlusion mask to attach to the Frame
        name: str
            If none, the occlusion mask will be attached without name (if possible). Otherwise if no other unnamed
            occlusion mask are attached to the frame, the mask will be added to the set of mask.
        """
        self._append_child("occlusion", occlusion, name)

    def __get_view__(self, min_disp=None, max_disp=None, cmap="nipy_spectral", reverse=False):
        assert all(dim not in self.names for dim in ["B", "T"]), "disparity should not have batch or time dimension"
        if cmap == "red2green":
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list("rg", ["r", "w", "g"], N=256)
        elif isinstance(cmap, str):
            cmap = matplotlib.cm.get_cmap(cmap)
        disp = self.unsigned().rename(None).permute([1, 2, 0]).as_numpy()
        disp = matplotlib.colors.Normalize(vmin=min_disp, vmax=max_disp, clip=True)(disp)
        if reverse:
            disp = 1 - disp
        disp_color = cmap(disp)[:, :, 0, :3]
        return View(disp_color)

    def _resize(self, size, **kwargs):
        """Resize disparity, but not its labels

        Parameters
        ----------
        size : tuple of float
            target size (H, W) in relative coordinates between 0 and 1

        Returns
        -------
        disp_resized : aloscene.Disparity
            resized version of disparity map
        """
        # resize disparity
        W_old = self.W
        disp_resized = super()._resize(size, **kwargs)
        W_new = disp_resized.W
        # rescale disparity
        sl_x = disp_resized.get_slices({"C": 0})
        labels = disp_resized.drop_children()
        disp_resized[sl_x] = disp_resized[sl_x] * W_new / W_old
        disp_resized.set_children(labels)
        return disp_resized

    def _hflip(self, **kwargs):
        """Flip disparity horizontally

        Returns
        -------
        disp_flipped : aloscene.Disparity
            horizontally flipped disparity map
        """
        disp_flipped = super()._hflip(**kwargs)

        if self.disp_format == "signed":
            disp_flipped = -1 * disp_flipped

        opposite = {"left": "right", "right": "left", None: None}
        disp_flipped.camera_side = opposite[disp_flipped.camera_side]
        return disp_flipped

    def unsigned(self):
        """
        Returns a copy of disparity map in unsigned disparity format
        """
        disp = self.clone()
        if disp.disp_format == "unsigned":
            return disp
        disp.disp_format = "unsigned"
        disp = torch.absolute(disp)
        return disp

    def signed(self, camera_side: str = None):
        """
        Returns a copy of disparity map in signed disparity format
        """
        disp = self.clone()
        if disp.disp_format == "signed":
            return disp
        camera_side = camera_side if camera_side is not None else disp.camera_side
        if camera_side is None:
            raise ValueError("Cannot convert disparity to signed format if `camera side` is None.")
        disp.disp_format = "signed"
        if camera_side == "left":
            disp = -1 * disp
        disp.camera_side = camera_side
        return disp

    def as_depth(
        self,
        baseline: float = None,
        focal_length: float = None,
        camera_side: float = None,
        camera_intrinsic: aloscene.CameraIntrinsic = None,
        max_depth=np.inf,
    ):
        """Return a Depth augmented tensor based on the given `baseline` & `focal_length`.

        Parameters
        ----------
        camera_side : str | None
            If created from a stereo camera, this information can optionally be used to convert
            this depth tensor into a disparity tensor. The `camera_side` is necessary to switch from unsigned to signed
            format once using a disparity tensor.
        baseline: float | None
            The `baseline` must be known to convert this disp tensor into a depth tensor.
            The `baseline` must be given either from from the current disp tensor or from this parameter.
        camera_intrinsic: aloscene.CameraIntrinsic
            CameraIntrinsic use to transform the disp map into depth map using the intrinsic focal
            length.
        """
        baseline = baseline if baseline is not None else self.baseline
        camera_side = camera_side if camera_side is not None else self.camera_side
        intrinsic = camera_intrinsic if camera_intrinsic is not None else self.cam_intrinsic

        if baseline is None:
            raise Exception(
                "Can't convert to disparity. The `baseline` must be given either from the current depth tensor or from the as_depth(baseline=...) method."
            )
        if intrinsic is None:
            err_msg = "The `camera_intrinsic` must be given either from the current disp tensor or from "
            err_msg += "the as_depth(camera_intrinsic=...) method."
            raise Exception(err_msg)

        disparity = self.unsigned().as_tensor()

        # unsqueeze on the Spatial H,W Dimension. On the dimension before "C", the focal_length is already supposed
        # to be aligned properly.
        focal_length = intrinsic.focal_length[..., 0:1].unsqueeze(-1).unsqueeze(-1)

        depth = baseline * focal_length / disparity
        depth = torch.clamp(depth, 0, max_depth)
        return aloscene.Depth(
            depth,
            baseline=self.baseline,
            camera_side=camera_side,
            cam_intrinsic=intrinsic,
            cam_extrinsic=self.cam_extrinsic,
            names=self.names,
            device=self.device,
        )
