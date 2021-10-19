from __future__ import annotations
import torch
from torch import Tensor

from typing import Union, Tuple

import aloscene
from aloscene import BoundingBoxes2D, CameraExtrinsic, CameraIntrinsic
from aloscene.renderer import View
from aloscene.labels import Labels
from aloscene.renderer.bbox3d import draw_3D_box

# from aloscene.camera_calib import CameraExtrinsic, CameraIntrinsic
from aloscene.utils.math_utils import get_y_rotation_matrixes, rotation_matrix_to_euler_angles

try:
    from aloscene.utils.rotated_iou.oriented_iou_loss import cal_giou_3d, cal_iou_3d

    import_error = False
except Exception as e:
    cal_giou_3d = None
    cal_iou_3d = None
    import_error = e


class BoundingBoxes3D(aloscene.tensors.AugmentedTensor):
    """
    Bounding Boxes 3D Tensor of shape (n, 7)
    of which the last dimension is : [xc, yc, zc, Dx, Dy, Dz, heading]

    - Coordinate xc, yc, zc of boxes' center
    - Boxes' dimension Dx, Dy, Dz along the 3 axis
    - Heading is the orientation by rotating around Y-axis.

    Coordinate system convention:

    - The X axis is positive to the right
    - The Y axis is positive downwards
    - The Z axis is positive forwards
    """

    @staticmethod
    def __new__(cls, x, labels: Union[dict, Labels] = None, names=("N", None), *args, **kwargs):
        tensor = super().__new__(cls, x, *args, names=names, **kwargs)
        assert tensor.shape[-1] == 7
        tensor.add_label("labels", labels, align_dim=["N"], mergeable=True)
        return tensor

    def __init__(self, x, *args, **kwargs):
        super().__init__(x)
        if import_error:
            print("==== ERROR ====")
            print(
                "It seems like sort_vertices custom cuda op needs to be built. \
                Please run `python setup.py install --user` from aloscene/utils/rotated_iou/cuda_op"
            )
            raise import_error

    @staticmethod
    def get_vertices_3d(boxes: BoundingBoxes3D) -> torch.Tensor:
        """Get 8 vertices for each boxes in x, y, z coordinates

        Parameters
        ----------
        boxes : BoundingBoxes3D
            Shape (n, 7)

        Returns
        -------
        torch.Tensor
            vertices in 3d
            Shape (n, 8, 3) for n boxes, 8 vertices, 3 coordinates [x, y, z]
        """
        boxes = boxes.as_tensor()
        zeros = torch.zeros((boxes.shape[0],)).to(boxes.device)
        centers = torch.unsqueeze(boxes[:, :3], 1)  # (n, 1, 3)
        Dx, Dy, Dz = boxes[:, 3], boxes[:, 4], boxes[:, 5]  # (n, )
        heading = boxes[:, 6]  # (n, )
        rot_matrix = get_y_rotation_matrixes(heading)
        vertices = torch.stack([-Dx / 2, -Dy / 2, -Dz / 2], dim=1)  # (n, 3)
        vertices = torch.unsqueeze(vertices, 0)  # (1, n, 3)
        vertices = torch.cat(
            [vertices, vertices + torch.stack([Dx, zeros, zeros], dim=1)], dim=0  # (1, n, 3) + (n, 3)
        )  # (2, n, 3)
        vertices = torch.cat(
            [vertices, vertices + torch.stack([zeros, Dy, zeros], dim=1)], dim=0  # (2, n, 3) + (n, 3)
        )  # (4, n, 3)
        vertices = torch.cat(
            [vertices, vertices + torch.stack([zeros, zeros, Dz], dim=1)], dim=0  # (4, n, 3) + (n, 3)
        )  # (8, n, 3)

        vertices = vertices.permute([1, 2, 0])
        vertices = rot_matrix @ vertices  # (n, 3, 3) @ (n, 3, 8) = (n, 3, 8)
        vertices = vertices.permute([0, 2, 1])
        return vertices + centers  # (n, 8, 3) + (n, 1, 3)

    @staticmethod
    def get_vertices_3d_proj(
        boxes: BoundingBoxes3D,
        cam_intrinsic: CameraIntrinsic,
        cam_extrinsic: CameraExtrinsic,
        include_vertice_behind_image_plan: bool = True,
    ) -> Tensor:
        """Get 8 vertices projected on image for each box 3d

        Parameters
        ----------
        boxes : BoundingBoxes3D
            Shape (n, 7)
        cam_intrinsic : aloscene.camera_calib.CameraIntrinsic
            Shape (3, 4)
        cam_extrinsic : CameraExtrinsic
            Shape (4, 4)
        include_vertice_behind_image_plan : bool, optional
            If True, returns all vertices from box behind the image plan (negative Z coordinates).
            If False, filter out those boxes
            By default True

        Returns
        -------
        Tensor
            Shape (m, 8, 2) for n boxes, 8 vertices, 2 coordinates x, y in pixel \n
            with m <= n
        """
        if isinstance(cam_intrinsic, CameraIntrinsic):
            cam_intrinsic = cam_intrinsic.as_tensor()
        if isinstance(cam_extrinsic, CameraExtrinsic):
            cam_extrinsic = cam_extrinsic.as_tensor()
        cam_extrinsic = cam_extrinsic.to(boxes.device)
        cam_intrinsic = cam_intrinsic.to(boxes.device)

        vertices = BoundingBoxes3D.get_vertices_3d(boxes)  # (n, 8, 3)
        vertices = vertices.transpose(1, 2)  # (n, 3, 8)
        ones = torch.unsqueeze(torch.ones((vertices.shape[0], 8)), dim=1).to(vertices.device)
        vertices = torch.cat([vertices, ones], dim=1)
        # vertices_2d = cam_intrinsic @ cam_extrinsic @ vertices # (n, 4, 8)
        vertices = cam_extrinsic @ vertices  # (n, 4, 8)
        if not include_vertice_behind_image_plan and vertices.shape[0] > 0:
            behind_plan_filter = vertices[:, 2, :].max(-1).values > 0
            vertices = vertices[behind_plan_filter]
        vertices_2d = cam_intrinsic @ vertices
        vertices_2d = vertices_2d[:, 0:2] / torch.unsqueeze(vertices_2d[:, 2], dim=1)  # (n, 2, 8)
        return vertices_2d.transpose(1, 2)

    @staticmethod
    def get_enclosing_box_2d(
        boxes: BoundingBoxes3D,
        cam_intrinsic: CameraIntrinsic,
        cam_extrinsic: CameraExtrinsic,
        frame_size: tuple[int, int],
        **kwargs,
    ) -> BoundingBoxes2D:
        """Get the 2d box enclosing the 3d box on image plan

        Parameters
        ----------
        boxes : BoundingBoxes3D
            Shape (n, 7)
        cam_intrinsic : CameraIntrinsic
            Shape (3, 4)
        cam_extrinsic : CameraExtrinsic
            Shape (4, 4)
        frame_size : tuple[int, int]
            (H, W)

        Returns
        -------
        BoundingBoxes2D
            Shape (n, 4). Format "xyxy" with absolute coordinates.
            This instance is populated with frame_size and the same `labels` as `boxes.labels`
        """
        if boxes.shape[0] > 0:
            vertices_2d = BoundingBoxes3D.get_vertices_3d_proj(boxes, cam_intrinsic, cam_extrinsic)  # (N, 8, 2)
            enclosing_box = torch.stack(
                [
                    vertices_2d.min(dim=1).values[:, 0],
                    vertices_2d.min(dim=1).values[:, 1],
                    vertices_2d.max(dim=1).values[:, 0],
                    vertices_2d.max(dim=1).values[:, 1],
                ],
                dim=1,
            )
        else:
            # empty tensor
            enclosing_box = torch.zeros((0, 4), device=boxes.device, dtype=boxes.dtype)
        return BoundingBoxes2D(
            enclosing_box, boxes_format="xyxy", absolute=True, frame_size=frame_size, labels=boxes.labels
        )

    @staticmethod
    def boxes_3d_hflip(boxes: BoundingBoxes3D, cam_extrinsic: Tensor, **kwargs):
        """Flip boxes 3d horizontally

        1. Bring boxes to camera refenrence frame using `cam_extrinsic`
        2. Flip x coordinates
        3. Rotate heading using Y-axis rotation of `cam_extrinsic` to get local heading w.r.t. camera. \
            Then flip the heading and finally rotate local heading back to global heading


        Parameters
        ----------
        boxes : BoundingBoxes3D
            shape (n, 7)
        cam_extrinsic : Tensor
            Shape (4, 4) or (t, 4, 4) of which the first dimension is "T"

        Returns
        -------
        BoundingBoxes3D
            flipped boungding boxes 3d of shape (n, 7)
        """
        names = boxes.names
        labels = boxes.labels
        boxes3d = boxes.as_tensor()
        if cam_extrinsic.names[0] == "T":
            # We suppose that cam_extrinsic is temporally consistent
            cam_extrinsic = cam_extrinsic.as_tensor()[0]
        elif len(cam_extrinsic.shape) == 2:
            cam_extrinsic = cam_extrinsic.as_tensor()
        else:
            raise Exception(
                f"cam_extrinsic should be rank 2, or 3 with names[0] == 'T', got {cam_extrinsic.names} {cam_extrinsic.shape}"
            )
        rotx, roty, rotz = rotation_matrix_to_euler_angles(cam_extrinsic[:3, :3])
        # inverse x coordinate in camera reference frame
        center3d = boxes3d[:, :3]
        center3d = torch.cat([center3d, torch.ones_like(center3d[:, 0:1])], dim=-1)
        cam_center3d = cam_extrinsic @ center3d.transpose(0, 1)  # -> (4, n)
        cam_center3d[0, :] = -cam_center3d[0, :]
        center3d = torch.linalg.inv(cam_extrinsic) @ cam_center3d
        center3d = center3d.transpose(0, 1)[:, :3]  # -> (n, 3)
        # dimension stays the same
        dimension = boxes3d[:, 3:6]
        # inverse heading
        heading = -boxes3d[:, 6:7] - 2 * roty
        # flipped boxes 3d
        flipped_boxes3d = torch.cat([center3d, dimension, heading], dim=-1)
        flipped_boxes3d = BoundingBoxes3D(flipped_boxes3d, labels=labels, names=names)
        return flipped_boxes3d

    def _hflip(self, *args, cam_extrinsic=None, **kwargs):
        return self.boxes_3d_hflip(self.clone(), cam_extrinsic)

    def _resize(self, *args, **kwargs):
        # Resize image does not change bbox3d
        return self.clone()

    def _crop(self, H_crop, W_crop, cam_intrinsic, cam_extrinsic, frame_size, **kwargs):
        """
        Filter out the bbox 3d out of frame

        Parameters
        ----------
        H_crop: tuple
            (start, end) between 0 and 1
        W_crop: tuple
            (start, end) between 0 and 1

        Returns
        -------
        cropped: BoundingBoxes3D
        """
        if cam_extrinsic.names[0] == "T":
            # We suppose that cam_extrinsic is temporally consistent
            cam_extrinsic = cam_extrinsic.as_tensor()[0]
        assert len(cam_extrinsic.shape) == 2
        if cam_intrinsic.names[0] == "T":
            # We suppose that cam_intrinsic is temporally consistent
            cam_intrinsic = cam_intrinsic.as_tensor()[0]
        assert len(cam_intrinsic.shape) == 2
        proj_box_2d = BoundingBoxes3D.get_enclosing_box_2d(self, cam_intrinsic, cam_extrinsic, frame_size)
        true_area = proj_box_2d.area()

        ymin = H_crop[0] * frame_size[0]
        ymax = H_crop[1] * frame_size[0]
        xmin = W_crop[0] * frame_size[1]
        xmax = W_crop[1] * frame_size[1]
        cropped_proj_box_2d = proj_box_2d.as_tensor()
        cropped_proj_box_2d[:, 0::2] = torch.clip(cropped_proj_box_2d[:, 0::2], min=xmin, max=xmax)
        cropped_proj_box_2d[:, 1::2] = torch.clip(cropped_proj_box_2d[:, 1::2], min=ymin, max=ymax)
        cropped_area = (cropped_proj_box_2d[:, 2] - cropped_proj_box_2d[:, 0]).mul(
            cropped_proj_box_2d[:, 3] - cropped_proj_box_2d[:, 1]
        )

        cropped_filter = cropped_area / true_area > 0.1
        return self.clone().rename_(None)[cropped_filter].reset_names()

    def _pad(self, *args, **kwargs):
        # Padding image does not change bbox 3d
        return self.clone()

    def bev_boxes(self) -> aloscene.OrientedBoxes2D:
        """
        Return the bird eye view boxes of shape (n, 4).

        The last dimension: (x, z, dx, dz, converted heading).
        Because XZ is in inversed order, `converted heading = - heading`
        """
        boxes = self.as_tensor()
        labels = None if self.labels is None else self.labels.clone()
        bv_boxes = torch.stack(
            [
                boxes[:, 0],  # x
                boxes[:, 2],  # z
                boxes[:, 3],  # dx
                boxes[:, 5],  # dz
                -boxes[:, 6],  # angle, -1*angle + pi/2 because x->z is the reversed order
            ],
            dim=-1,
        )
        return aloscene.OrientedBoxes2D(bv_boxes, labels=labels, names=self.names)

    @staticmethod
    def iou3d(
        boxes1: BoundingBoxes3D, boxes2: BoundingBoxes3D, ret_union=False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Calculate IoU 3D for 2 aligned sets of boxes in order

        This function works only with `boxes1` and `boxes2` in CUDA device.

        Parameters
        ----------
        boxes1 : BoundingBoxes3D
            Shape (n, 7)
        boxes2 : BoundingBoxes3D
            Shape (n, 7)
        ret_union : bool, optional
            If True, return also union volume, by default False

        Returns
        -------
        Tensor or tuple of (Tensor, Tensor)
            IoU 3D, of shape (n,)
            Union volume, of shape (n,) (if `ret_union` True)
        """
        assert boxes1.shape[0] == boxes2.shape[0], "Two sets must have the same number of boxes"
        if boxes1.shape[0] == 0:
            iou3d = torch.zeros((0,), device=boxes1.device)
            union3d = torch.zeros((0,), device=boxes1.device)
        else:
            if isinstance(boxes1, BoundingBoxes3D):
                boxes1 = boxes1.as_tensor()
            if isinstance(boxes2, BoundingBoxes3D):
                boxes2 = boxes2.as_tensor()
            iou3d, _, _, _, union3d = cal_iou_3d(boxes1[None], boxes2[None], verbose=True)
            iou3d = iou3d[0]
            union3d = union3d[0]
        if ret_union:
            return iou3d, union3d
        else:
            return iou3d

    def iou3d_with(self, boxes2: BoundingBoxes3D, ret_union=False):
        """Calculate IoU 3D for 2 aligned sets of boxes in order

        This function works only with `self` and `boxes2` in CUDA device.

        Parameters
        ----------
        boxes2 : BoundingBoxes3D
            Shape (n, 7)
        ret_union : bool, optional
            If True, return also union volume, by default False

        Returns
        -------
        Tensor or tuple of (Tensor, Tensor)
            IoU 3D, of shape (n,)
            Union volume, of shape (n,) (if `ret_union` True)
        """
        return self.iou3d(self, boxes2, ret_union)

    @staticmethod
    def giou3d(
        boxes1: Union[torch.Tensor, BoundingBoxes3D],
        boxes2: Union[torch.Tensor, BoundingBoxes3D],
        enclosing_type="smallest",
        ret_iou3d=False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Calculate GIoU 3D for 2 aligned sets of boxes in order

        This function works only with `boxes1` and `boxes2` in CUDA device.

        Parameters
        ----------
        boxes1 : BoundingBoxes3D or torch.Tensor
            Shape (n, 7)
        boxes2 : BoundingBoxes3D or torch.Tensor
            Shape (n, 7)
        enclosing_type : str, optional
            Choose the algorithm for finding enclosing box :
                aligned      # simple and naive. bad performance. fastest
                pca          # approximated smallest box. slightly worse performance.
                smallest     # [default]. brute force. smallest box. best performance. slowest
        ret_iou3d : bool, optional
            If True, return also IoU 3D, by default False

        Returns
        -------
        Tensor or tuple of (Tensor, Tensor)
            GIoU 3D, of shape (n,)
            IoU 3D, of shape (n,) (if `ret_iou3d` True)
        """
        assert boxes1.shape[0] == boxes2.shape[0], "Two sets must have the same number of boxes"
        if boxes1.shape[0] == 0:
            giou = torch.zeros((0,), device=boxes1.device)
            iou = torch.zeros((0,), device=boxes1.device)
        else:
            if isinstance(boxes1, BoundingBoxes3D):
                boxes1 = boxes1.as_tensor()
            if isinstance(boxes2, BoundingBoxes3D):
                boxes2 = boxes2.as_tensor()
            giou, iou = cal_giou_3d(boxes1[None], boxes2[None], enclosing_type)
            giou = giou[0]
            iou = iou[0]
        if ret_iou3d:
            return giou, iou
        else:
            return giou

    def giou3d_with(
        self, boxes2: BoundingBoxes3D, enclosing_type="smallest", ret_iou3d=False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Calculate GIoU 3D for 2 aligned sets of boxes in order

        This function works only with `self` and `boxes2` in CUDA device.

        Parameters
        ----------
        boxes2 : BoundingBoxes3D
            Shape (n, 7)
        enclosing_type : str, optional
            Choose the algorithm for finding enclosing box :
                aligned      # simple and naive. bad performance. fastest
                pca          # approximated smallest box. slightly worse performance.
                smallest     # [default]. brute force. smallest box. best performance. slowest
        ret_iou3d : bool, optional
            If True, return also IoU 3D, by default False

        Returns
        -------
        Tensor or tuple of (Tensor, Tensor)
            GIoU 3D, of shape (n,)
            IoU 3D, of shape (n,) (if `ret_iou3d` True)
        """
        return self.giou3d(self, boxes2, enclosing_type, ret_iou3d)

    def get_view(self, frame: aloscene.Frame, size: tuple = None, mode: str = "3D", **kwargs) -> View:
        """Create a View instance from a Frame

        Parameters
        ----------
        frame : aloscene.Frame
            The Frame must contain cam_intrinsic and cam_extrinsic matrices
        size : tuple, optional
            Desired size of the view, by default None.
        mode : str, optional
            View mode, either "3D" or "2D", by default "3D"

        Returns
        -------
        View
        """
        assert frame.cam_intrinsic is not None
        assert frame.cam_extrinsic is not None
        frame_size = frame.HW
        cam_intrinsic = frame.cam_intrinsic.as_tensor()
        cam_extrinsic = frame.cam_extrinsic.as_tensor()

        if mode == "3D":
            # Get vertices proj
            vertices_3d_proj = (
                self.get_vertices_3d_proj(self, cam_intrinsic, cam_extrinsic, include_vertice_behind_image_plan=True)
                .cpu()
                .numpy()
            )
            # Draw bbox 3d
            frame = frame.norm01().cpu().rename(None).permute([1, 2, 0]).detach().contiguous().numpy()
            draw_3D_box(frame, vertices_3d_proj)
            return View(frame, **kwargs)
        elif mode == "2D":
            enclosing_2d = self.get_enclosing_box_2d(self, cam_intrinsic, cam_extrinsic, frame_size)
            return enclosing_2d.get_view(frame, size)
        elif mode == "BEV":
            raise NotImplementedError()
        else:
            Exception("view mode {mode} is not supported")
