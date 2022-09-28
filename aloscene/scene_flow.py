import aloscene
from aloscene import Depth, CameraIntrinsic, Mask, Flow
from aloscene.io.flow import load_scene_flow
from typing import Union
import torch
import torch.nn.functional as F
from aloscene.renderer import View
from aloscene.utils.flow_utils import flow_to_color


class SceneFlow(aloscene.tensors.SpatialAugmentedTensor):
    """
    Scene flow map

    Parameters
    ----------
    x : str or tensor or ndarray
        load scene flow from a numpy file
    """

    @staticmethod
    def __new__(cls, x, occlusion: Union[Mask, None] = None, *args, names=("C", "H", "W"), **kwargs):
        if isinstance(x, str):
            # load flow from path
            x = load_scene_flow(x)
            names = ("C", "H", "W")

        tensor = super().__new__(cls, x, *args, names=names, **kwargs)
        tensor.add_child("occlusion", occlusion, align_dim=["B", "T"], mergeable=True)
        return tensor

    def __init__(self, x, *args, **kwargs):
        super().__init__(x)

    def __get_view__(self, clip_flow=None, convert_to_bgr=False, magnitude_max=None, axes="xx"):
        assert all(dim not in self.names for dim in ["B", "T"]), "flow should not have batch or time dimension"
        axe_one = ord(axes[0]) - ord("x")
        axe_two = ord(axes[1]) - ord("x")
        assert len(axes) == 2, "axes should be of length 2"
        assert axe_one >= 0 and axe_one <= 2 and axe_two >= 0 and axe_two <= 2, "axes should be x, y or z"
        flow = (
            self.rename(None)
            .squeeze()
            .permute([1, 2, 0])
            .as_numpy()[:, :, [axe_one, axe_two]]
        )
        assert flow.ndim == 3 and flow.shape[-1] == 2, f"wrong flow shape:{flow.shape}"
        flow_color = flow_to_color(flow, clip_flow, convert_to_bgr, magnitude_max) / 255
        return View(flow_color)

    @classmethod
    def from_optical_flow(
        cls,
        optical_flow: Flow,
        depth: Depth,
        next_depth: Depth,
        intrinsic: CameraIntrinsic,
        sampling: str = "bilinear",
    ):
        """Create scene flow from optical flow, depth a T, depth at T + 1 and the intrinsic

        Parameters
        ----------
        optical flow: aloscene.Flow
            The optical flow at T.
        depth: aloscene.Depth
            The depth at T.
        next_depth: aloscene.Depth
            The depth at T + 1
        intrinsic : aloscene.CameraIntrinsic
            The intrinsic of the image at T.
        sampling: str
            The sampling method to use for the scene flow.
        """
        has_batch = "B" in optical_flow.names

        if optical_flow.names != depth.names or optical_flow.names != next_depth.names:
            raise ValueError("The optical flow, depth and next_depth must have the same names")

        if optical_flow.names != ("C", "H", "W") and optical_flow.names != ("B", "C", "H", "W"):
            raise ValueError("The optical flow must have the names (C, H, W) or (B, C, H, W)")

        # Artifical batch dimension
        optical_flow = optical_flow.batch()
        depth = depth.batch()
        next_depth = next_depth.batch()

        H, W = depth.HW
        B = depth.shape[0]

        # Compute the point cloud at T and T + 1
        start_vector = depth.as_points3d(intrinsic).as_tensor().reshape(-1, H, W, 3).permute(0, 3, 1, 2)
        next_vector = next_depth.as_points3d(intrinsic).as_tensor().reshape(-1, H, W, 3).permute(0, 3, 1, 2)

        # Compute the position of the point cloud at T + 1
        y_coords, x_coords = torch.meshgrid(torch.arange(H), torch.arange(W))
        new_x = x_coords + optical_flow.as_tensor()[:, 0, :, :]
        new_y = y_coords + optical_flow.as_tensor()[:, 1, :, :]

        # Normalize the coordinates bettwen -1 and 1 and create the new points coordinates
        new_x = new_x / W * 2 - 1
        new_y = new_y / H * 2 - 1
        new_coords = torch.stack([new_x, new_y], dim=3)

        # Move the point cloud at T + 1 to the new position
        end_vector = F.grid_sample(next_vector, new_coords, mode=sampling, padding_mode="zeros", align_corners=True)

        # Compute the scene flow
        scene_flow_vector = end_vector - start_vector

        # Create the occlusion mask if needed
        occlusion = None
        if optical_flow.occlusion is not None or depth.occlusion is not None or next_depth.occlusion is not None:
            occlusion = torch.zeros(B, H, W, dtype=torch.bool)

            # Add depth and optical flow occlusion to main occlusion mask
            if optical_flow.occlusion is not None:
                occlusion = occlusion | optical_flow.occlusion.as_tensor().bool()
            if depth.occlusion is not None:
                occlusion = occlusion | depth.occlusion.as_tensor().bool()
            if next_depth.occlusion is not None:
                next_depth_tensor = next_depth.occlusion.as_tensor().bool()

                # Use of 'not' needed because the grid_sample has padding_mode="zeros" and
                # the 0 from this function mean that the pixel is occluded
                next_depth_tensor = ~next_depth_tensor

                # Move the occlusion mask like the scene flow to check if occluded pixel are used in the calculation
                moved_occlusion = F.grid_sample(
                    next_depth_tensor.float(), new_coords, mode=sampling, padding_mode="zeros", align_corners=True
                )

                # Sometimes moved_occlusion is not exactly 1 even if the pixels around are not occluded
                moved_occlusion = ~(moved_occlusion >= 0.99999)

                # Fusion of the 2 occlusion mask
                occlusion = occlusion | moved_occlusion

        # Remove the artificial batch dimension
        if not has_batch:
            scene_flow_vector = scene_flow_vector.squeeze(0)
            optical_flow = optical_flow.squeeze(0)
            occlusion = occlusion.squeeze(0) if occlusion is not None else None

        # Create the scene flow object
        tensor = cls(
            scene_flow_vector,
            names=("B", "C", "H", "W") if has_batch else ("C", "H", "W"),
            occlusion=None
            if occlusion is None
            else Mask(occlusion, names=("B", "C", "H", "W") if has_batch else ("C", "H", "W")),
        )
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
