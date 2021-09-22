import aloscene
from aloscene import Mask
from aloscene.io.flow import load_flow
from aloscene.renderer import View
from aloscene.utils.flow_utils import flow_to_color


class Flow(aloscene.tensors.SpatialAugmentedTensor):
    """Optical Flow Map.

    Parameters
    ----------
    x : str or tensor
        path to a flow file (".flo") or flow tensor
    occlusion : aloscene.Mask
        occlusion mask attached to flow map
    """

    @staticmethod
    def __new__(cls, x, occlusion: Mask = None, *args, names=("C", "H", "W"), **kwargs):
        if isinstance(x, str):
            # load flow from path
            x = load_flow(x)
            names = ("C", "H", "W")
        tensor = super().__new__(cls, x, *args, names=names, **kwargs)
        tensor.add_label("occlusion", occlusion, align_dim=["B", "T"], mergeable=True)
        return tensor

    def __init__(self, x, *args, **kwargs):
        super().__init__(x)

    def append_occlusion(self, occlusion: Mask, name: str = None):
        """Attach an occlusion mask to the frame.

        Parameters
        ----------
        occlusion: aloscene.Mask
            Occlusion mask to attach to the Frame
        name: str
            If none, the occlusion mask will be attached without name (if possible). Otherwise if no other unnamed
            occlusion mask are attached to the frame, the mask will be added to the set of mask.
        """
        self._append_label("occlusion", occlusion, name)

    def __get_view__(self, clip_flow=None, convert_to_bgr=False, magnitude_max=None):
        assert all(dim not in self.names for dim in ["B", "T"]), "flow should not have batch or time dimension"
        flow = self.rename(None).squeeze().permute([1, 2, 0]).detach().cpu().contiguous().numpy()
        assert flow.ndim == 3 and flow.shape[-1] == 2, f"wrong flow shape:{flow.shape}"
        flow_color = flow_to_color(flow, clip_flow, convert_to_bgr, magnitude_max) / 255
        return View(flow_color)

    def _resize(self, size, **kwargs):
        """Resize Flow, but not its labels.

        Parameters
        ----------
        size : tuple of float
            target size (H, W) in relative coordinates between 0 and 1

        Returns
        -------
        flow_resized : aloscene.Flow
            resized version of flow map
        """
        H_old, W_old = self.H, self.W
        flow_resized = super()._resize(size, **kwargs)
        H_new, W_new = flow_resized.H, flow_resized.W
        # scale flow coordinates because they are expressed in pixel units
        sl_x = flow_resized.get_slices({"C": 0})  # slice for x coord. of flow vector
        sl_y = flow_resized.get_slices({"C": 1})  # slice for y coord. of flow vector
        labels = flow_resized.drop_labels()
        flow_resized[sl_x] = flow_resized[sl_x] * W_new / W_old
        flow_resized[sl_y] = flow_resized[sl_y] * H_new / H_old
        flow_resized.set_labels(labels)
        return flow_resized

    def _hflip(self, **kwargs):
        """Flip flow horizontally.

        Returns
        -------
        flipped_flow : aloscene.Flow
            horizontally flipped flow map
        """
        flow_flipped = super()._hflip(**kwargs)
        # invert x axis of flow vector
        labels = flow_flipped.drop_labels()
        sl_x = flow_flipped.get_slices({"C": 0})
        flow_flipped[sl_x] = -1 * flow_flipped[sl_x]
        flow_flipped.set_labels(labels)
        return flow_flipped

    def _vflip(self, **kwargs):
        """Flip flow vertically.

        Returns
        -------
        flipped_flow : aloscene.Flow
            vertically flipped flow map
        """
        flow_flipped = super()._vflip(**kwargs)
        # invert y axis of flow vector
        labels = flow_flipped.drop_labels()
        sl_y = flow_flipped.get_slices({"C": 1})
        flow_flipped[sl_y] = -1 * flow_flipped[sl_y]
        flow_flipped.set_labels(labels)
        return flow_flipped
