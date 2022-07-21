import aloscene
from aloscene import Depth, CameraIntrinsic, Flow, Mask
from typing import Union
import numpy as np
import torch
import torch.nn.functional as F


def load_scene_flow(path: str) -> torch.Tensor:
    with open(path, "rb") as file:
        data = np.load(file)
        return torch.from_numpy(data)


class SceneFlow(aloscene.tensors.SpatialAugmentedTensor):
    """
    Scene flow map

    Parameters
    ----------
    x : str
        load scene flow from a numpy file
    """

    @staticmethod
    def __new__(cls, x, occlusion: Mask = None, *args, names=("C", "H", "W"), **kwargs):
        if isinstance(x, str):
            # load flow from path
            x = load_scene_flow(x)
            names = ("C", "H", "W")

        tensor = super().__new__(cls, x, *args, names=names, **kwargs)
        tensor.add_child("occlusion", occlusion, align_dim=["B", "T"], mergeable=True)
        return tensor

    def __init__(self, x, *args, **kwargs):
        super().__init__(x)

    @classmethod
    def from_optical_flow(
        cls,
        optical_flow: Flow,
        depth: Depth,
        next_depth: Depth,
        intrinsic: CameraIntrinsic,
    ):
        """Create scene flow from optical flow, depth a T, depth at T + 1 and the intrinsic

        Parameters
        ----------
        optical flow: aloscene.Flow
            The optical flow at T.
        depth: aloscene.Depth
            The depth at T."
        next_depth: aloscene.Depth
            The depth at T + 1
        intrinsic : aloscene.CameraIntrinsic
            The intrinsic of the image at T.
        """
        start_vector = (
            depth.as_points3d(camera_intrinsic=intrinsic)
            .cpu()
            .numpy()
            .reshape((depth.H, depth.W, 3))
        )

        next_vector = (
            next_depth.as_points3d(camera_intrinsic=intrinsic)
            .cpu()
            .numpy()
            .reshape((next_depth.H, next_depth.W, 3))
        )

        ceilflow = np.ceil(optical_flow.numpy()).astype(int)
        floorflow = np.floor(optical_flow.numpy()).astype(int)

        y_coords, x_coords = np.mgrid[: depth.H, : depth.W]

        ceil_new_x = np.clip(x_coords + ceilflow[:, :, 0], 0, depth.W - 1)
        ceil_new_y = np.clip(y_coords + ceilflow[:, :, 1], 0, depth.H - 1)
        floor_new_x = np.clip(x_coords + floorflow[:, :, 0], 0, depth.W - 1)
        floor_new_y = np.clip(y_coords + floorflow[:, :, 1], 0, depth.H - 1)

        scene_flow_vector = (
            next_vector[ceil_new_y, ceil_new_x, :]
            + next_vector[floor_new_y, ceil_new_x, :]
            + next_vector[ceil_new_y, floor_new_x, :]
            + next_vector[floor_new_y, floor_new_x, :]
        ) / 4 - start_vector

        scene_flow_vector = np.transpose(scene_flow_vector, (2, 0, 1))
        masked_points = Mask(np.isfinite(scene_flow_vector).all(0), names=("H", "W"))

        tensor = cls(scene_flow_vector)
        tensor.append_mask(masked_points)
        tensor.append_occlusion(optical_flow.occlusion.clone(), "occlusion")
        return tensor

    def append_occlusion(self, occlusion: Mask, name: Union[str, None] = None):
        """Attach an occlusion mask to the scene flow.

        Parameters
        ----------
        occlusion: aloscene.Mask
            Occlusion mask to attach to the Scene Flow
        name: str
            If none, the occlusion mask will be attached without name (if possible). Otherwise if no other unnamed
            occlusion mask are attached to the scene flow, the mask will be added to the set of mask.
        """
        self._append_child("occlusion", occlusion, name)

    def _hflip(self, **kwargs):
        """Flip scene flow horizontally.

        Returns
        -------
        flipped_scene_flow : aloscene.SceneFlow
            horizontally flipped scene flow map
        """
        flow_flipped = super()._hflip(**kwargs)
        # invert x axis of flow vector
        labels = flow_flipped.drop_children()
        sl_x = flow_flipped.get_slices({"C": 0})
        flow_flipped[sl_x] = -1 * flow_flipped[sl_x]
        flow_flipped.set_children(labels)
        return flow_flipped

    def _vflip(self, **kwargs):
        """Flip scene flow vertically.

        Returns
        -------
        flipped_scene_flow : aloscene.SceneFlow
            vertically flipped scene flow map
        """
        flow_flipped = super()._vflip(**kwargs)
        # invert y axis of flow vector
        labels = flow_flipped.drop_children()
        sl_y = flow_flipped.get_slices({"C": 1})
        flow_flipped[sl_y] = -1 * flow_flipped[sl_y]
        flow_flipped.set_children(labels)
        return flow_flipped
