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
    camera_side : {'left', 'right', None}
        side of the camera. Default None.
        Necessary to switch from unsigned to signed format.
    png_negate: bool
        if true, the sign of disparity is reversed when loaded from file.
        this parameter should be explicitely set every time a .png file is used.
    """

    @staticmethod
    def __new__(
        cls, x, occlusion: Mask = None, disp_format="unsigned", camera_side=None, png_negate=None, *args, **kwargs
    ):
        if isinstance(x, str):
            x = load_disp(x, png_negate)
            kwargs["names"] = ("C", "H", "W")
        tensor = super().__new__(cls, x, *args, **kwargs)
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

    def __get_view__(self):
        assert all(dim not in self.names for dim in ["B", "T"]), "disparity should not have batch or time dimension"
        cmap = matplotlib.cm.get_cmap("nipy_spectral")
        disp = self.rename(None).permute([1, 2, 0]).detach().contiguous().numpy()
        disp = matplotlib.colors.Normalize()(disp)
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
        labels = disp_resized.drop_labels()
        disp_resized[sl_x] = disp_resized[sl_x] * W_new / W_old
        disp_resized.set_labels(labels)
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

    def signed(self):
        """
        Returns a copy of disparity map in signed disparity format
        """
        disp = self.clone()
        if disp.disp_format == "signed":
            return disp
        if disp.camera_side is None:
            raise ValueError("Cannot convert disparity to signed format if `camera side` is None.")
        disp.disp_format = "signed"
        if disp.camera_side == "left":
            disp = -1 * disp
        return disp
