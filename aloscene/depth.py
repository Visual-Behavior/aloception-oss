import matplotlib
import torch

import aloscene
from aloscene import Mask
from aloscene.renderer import View
import numpy as np


class Depth(aloscene.tensors.SpatialAugmentedTensor):
    """
    Depth map

    Parameters
    ----------
    x: tensor or str
        loaded Depth tensor or path to the Depth file from which Depth will be loaded.
    occlusion : aloscene.Mask
        Occlusion mask for this Depth map. Default value : None.
    disp_format : {'signed'|'unsigned'}
        If unsigned, Depth is interpreted as a distance (positive value) in pixels.
        If signed, Depth is interpreted as a relative offset (values can be negative).
    camera_side : {'left', 'right', None}
        side of the camera. Default None.
        Necessary to switch from unsigned to signed format.
    png_negate: bool
        if true, the sign of Depth is reversed when loaded from file.
        this parameter should be explicitely set every time a .png file is used.
    """

    @staticmethod
    def __new__(
        cls,
        x,
        occlusion: Mask = None,
        disp_format="unsigned",
        camera_side=None,
        png_negate=None,
        *args,
        names=("C", "H", "W"),
        **kwargs
    ):
        if isinstance(x, str):
            x = load_disp(x, png_negate)
            names = ("C", "H", "W")
        tensor = super().__new__(cls, x, *args, names=names, **kwargs)
        tensor.add_label("occlusion", occlusion, align_dim=["B", "T"], mergeable=True)
        tensor.add_property("disp_format", disp_format)
        tensor.add_property("camera_side", camera_side)
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
        self._append_label("occlusion", occlusion, name)

    def __get_view__(self, vmax=200):
        assert all(dim not in self.names for dim in ["B", "T"]), "Depth should not have batch or time dimension"
        cmap = matplotlib.cm.get_cmap("nipy_spectral")
        depth = self.rename(None).permute([1, 2, 0]).detach().contiguous().numpy()

        depth = vmax - np.clip(depth, 0, vmax)
        depth = matplotlib.colors.Normalize(vmax=vmax)(depth)
        depth_color = cmap(depth)[:, :, 0, :3]
        return View(depth_color)

    def _resize(self, size, **kwargs):
        """Resize Depth, but not its labels

        Parameters
        ----------
        size : tuple of float
            target size (H, W) in relative coordinates between 0 and 1

        Returns
        -------
        disp_resized : aloscene.Depth
            resized version of Depth map
        """
        # resize Depth
        W_old = self.W
        disp_resized = super()._resize(size, **kwargs)
        W_new = disp_resized.W
        # rescale Depth
        sl_x = disp_resized.get_slices({"C": 0})
        labels = disp_resized.drop_labels()
        disp_resized[sl_x] = disp_resized[sl_x] * W_new / W_old
        disp_resized.set_labels(labels)
        return disp_resized

    def _hflip(self, **kwargs):
        """Flip Depth horizontally

        Returns
        -------
        disp_flipped : aloscene.Depth
            horizontally flipped Depth map
        """
        disp_flipped = super()._hflip(**kwargs)

        if self.disp_format == "signed":
            disp_flipped = -1 * disp_flipped

        opposite = {"left": "right", "right": "left", None: None}
        disp_flipped.camera_side = opposite[disp_flipped.camera_side]
        return disp_flipped

    def unsigned(self):
        """
        Returns a copy of Depth map in unsigned Depth format
        """
        disp = self.clone()
        if disp.disp_format == "unsigned":
            return disp
        disp.disp_format = "unsigned"
        disp = torch.absolute(disp)
        return disp

    def signed(self):
        """
        Returns a copy of Depth map in signed Depth format
        """
        disp = self.clone()
        if disp.disp_format == "signed":
            return disp
        if disp.camera_side is None:
            raise ValueError("Cannot convert Depth to signed format if `camera side` is None.")
        disp.disp_format = "signed"
        if disp.camera_side == "left":
            disp = -1 * disp
        return disp
