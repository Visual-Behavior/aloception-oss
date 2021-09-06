import aloscene
from aloscene.renderer import View
from aloscene.io.mask import load_mask


class Mask(aloscene.tensors.SpatialAugmentedTensor):
    """Binary or Float Mask

    Parameters
    ----------
    x :
        path to the mask file (png) or tensor (values between 0. and 1.)
    """

    @staticmethod
    def __new__(cls, x, *args, **kwargs):
        # Load frame from path
        if isinstance(x, str):
            x = load_mask(x)
            kwargs["names"] = ("C", "H", "W")
        tensor = super().__new__(cls, x, *args, **kwargs)
        return tensor

    def __init__(self, x, *args, **kwargs):
        super().__init__(x)

    def __get_view__(self, title=None):
        """Create a view of the frame"""
        assert self.names[0] != "T" and self.names[1] != "B"
        frame = self.cpu().rename(None).permute([1, 2, 0]).detach().contiguous().numpy()
        view = View(frame, title=title)
        return view
