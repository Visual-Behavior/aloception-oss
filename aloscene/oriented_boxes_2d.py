# from __future__ import annotations
import torch
from torch import Tensor

from typing import Union
import numpy as np
import cv2

import aloscene
from aloscene.renderer import View
from aloscene.labels import Labels

try:
    from aloscene.utils.rotated_iou.box_intersection_2d import oriented_box_intersection_2d
    from aloscene.utils.rotated_iou.oriented_iou_loss import cal_giou

    import_error = False
except Exception as e:
    oriented_box_intersection_2d = None
    cal_giou = None
    import_error = e


class OrientedBoxes2D(aloscene.tensors.AugmentedTensor):
    """Oriented Boxes 2D is defined by [x, y, w, h, theta] in which:
    - x, y: center coordinates
    - w, h: width, height
    - theta: rotation angle
    """

    @staticmethod
    def __new__(
        cls,
        x,
        absolute=True,
        frame_size=None,
        labels: Union[dict, Labels, None] = None,
        names=("N", None),
        *args,
        **kwargs,
    ):
        tensor = super().__new__(cls, x, *args, names=names, **kwargs)

        assert x.shape[-1] == 5, "The last dimension should be [x, y, w, h, theta]"
        # Add label
        tensor.add_child("labels", labels, align_dim=["N"], mergeable=True)

        tensor.add_property("absolute", absolute)
        tensor.add_property("frame_size", frame_size)

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

    def append_labels(self, labels: Labels, name: Union[str, None] = None):
        """Attach a set of labels to the boxes.

        Parameters
        ----------
        labels : aloscene.Labels
            Set of labels to attached to the frame
        name : str
            If none, the label will be attached without name (if possible). Otherwise if no other unnamed
            labels are attached to the frame, the labels will be added to the set of labels.
        """
        self._append_child("labels", labels, name)

    def corners(self) -> torch.Tensor:
        """Get corners in x, y coordinates

        Corners are in counter-clockwise order, started from the top right corner

        Returns
        -------
        torch.Tensor
            shape (n, 4, 2) for n boxes, 4 vertices, 2 x-y coordinates
        """
        boxes = self.as_tensor()
        x, y, w, h, alpha = boxes.split(1, -1)
        x4 = torch.tensor([0.5, -0.5, -0.5, 0.5], device=self.device) * w
        y4 = torch.tensor([0.5, 0.5, -0.5, -0.5], device=self.device) * h
        corners = torch.stack([x4, y4], dim=-1)
        sin = torch.sin(alpha)
        cos = torch.cos(alpha)
        R = torch.cat([cos, -sin, sin, cos], dim=1).reshape(-1, 2, 2)
        corners = corners @ R.transpose(1, 2)
        corners[..., 0] += x
        corners[..., 1] += y
        return corners

    GLOBAL_COLOR_SET = np.random.uniform(0, 1, (300, 3))

    def get_view(
        self,
        frame: Union[Tensor, None] = None,
        size: Union[tuple, None] = None,
        color: Union[tuple, None] = None,
        labels_set: Union[str, None] = None,
        **kwargs,
    ):
        """Create a view of the boxes in a frame

        Parameters
        ----------
        frame : aloscene.Frame
            Tensor of type Frame to display the boxes on.
            If the frame is None, a blank frame will be created.
        size : (tuple)
            (height, width) Desired size of the view. None by default
        color : tuple
            (R, G, B). None by default, a random color will be choosen.
        labels_set : str
            If provided, the boxes will rely on this label set to display the boxes color,
            and `color` parameter will be ignore.
            If not provided while the boxes have multiple labels set,
            the boxes will be display with the same color which can be set with `color` param.
        """
        if frame is not None:
            if len(frame.shape) > 3:
                raise Exception(f"Expect image of shape c,h,w. Found image with shape {frame.shape}")
            assert isinstance(frame, aloscene.Frame)
        else:
            assert (
                self.frame_size is not None or size is not None
            ), "Either `size` or `self.frame_size` must be defined."
            boxes = self.clone()
            if boxes.frame_size is None:
                boxes.frame_size = size
            size = size or boxes.frame_size
            frame = torch.zeros(3, int(size[0]), int(size[1]))
            frame = aloscene.Frame(frame, names=("C", "H", "W"), normalization="01")
        boxes_abs = boxes.abs_pos(frame.HW)
        # Get an imave with values between 0 and 1
        frame = frame.norm01().cpu().rename(None).permute([1, 2, 0]).detach().contiguous().numpy()
        # === Draw bounding boxes
        # Try to retrieve the associated label ID (if any)
        labels = boxes_abs.labels if isinstance(boxes_abs.labels, aloscene.Labels) else [None] * len(boxes_abs)
        if labels_set is not None and not isinstance(boxes_abs.labels, dict):
            raise Exception(
                f"Trying to display a set of boxes labels ({labels_set}) \
                    while the boxes do not have multiple set of labels"
            )
        elif labels_set is not None and isinstance(boxes_abs.labels, dict) and labels_set not in boxes_abs.labels:
            raise Exception(
                f"Trying to display a set of boxes labels ({labels_set}) while the boxes no not have this set. \
                    Avaiable set ("
                + [key for key in boxes_abs.labels]
                + ") "
            )
        elif labels_set is not None:
            labels = boxes_abs.labels[labels_set]
            assert labels.encoding == "id"

        boxes_corners = boxes_abs.corners().detach().cpu().numpy().astype(np.int0)
        for corners, label in zip(boxes_corners, labels):
            color = color if color is not None else (0, 1, 0)
            top_left = corners.min(0)
            if label is not None:
                color = self.GLOBAL_COLOR_SET[int(label) % len(self.GLOBAL_COLOR_SET)]
                cv2.putText(
                    frame,
                    str(int(label)),
                    (int(top_left[0]), int(top_left[1])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )
            cv2.drawContours(frame, [corners], 0, color, 2)
        # Return the view to display
        return View(frame, **kwargs)

    @staticmethod
    def boxes2abspos(tensor, frame_size: tuple):
        """Get a new OrientedBoxes2D Tensor with absolute position \
        relative to the given `frame_size` (H, W)

        Parameters
        ----------
        tensor: OrientedBoxes2D
        frame_size : tuple
            (Height, Width)

        Returns
        -------
        OrientedBoxes2D
            boxes in absolute coordinates
        """
        # Back to relative before to get the absolute pos
        if tensor.absolute and frame_size != tensor.frame_size:
            tensor.div(
                torch.tensor([frame_size[1], frame_size[0], frame_size[1], frame_size[0], 1.0], device=tensor.device)
            )
            tensor.absolute = False
        if not tensor.absolute:
            tensor = tensor.mul(
                torch.tensor([frame_size[1], frame_size[0], frame_size[1], frame_size[0], 1.0], device=tensor.device)
            )
            tensor.frame_size = frame_size
            tensor.absolute = True
        elif tensor.absolute and frame_size == tensor.frame_size:
            pass
        else:
            raise Exception("boxes2abspos no handled error")
        return tensor

    def abs_pos(self, frame_size: tuple):
        """Get a new OrientedBoxes2D Tensor with absolute position \
        relative to the given `frame_size` (H, W)

        Parameters
        ----------
        frame_size : tuple
            (Height, Width)

        Returns
        -------
        OrientedBoxes2D
            boxes in absolute coordinates
        """
        return self.boxes2abspos(self.clone(), frame_size)

    @staticmethod
    def boxes2relpos(tensor):
        """Get a new OrientedBoxes2D Tensor with relative position
        (between 0 and 1)

        Parameters
        ----------
        tensor : OrientedBoxes2D

        Returns
        -------
        OrientedBoxes2D
        """
        if tensor.absolute:
            tensor = tensor.div(
                torch.tensor(
                    [tensor.frame_size[1], tensor.frame_size[0], tensor.frame_size[1], tensor.frame_size[0], 1.0],
                    device=tensor.device,
                )
            )

        elif not tensor.absolute:
            pass
        tensor.absolute = False
        return tensor

    def rel_pos(self):
        """Get a new OrientedBoxes2D Tensor with relative position
        (between 0 and 1)

        Returns
        -------
        OrientedBoxes2D
        """
        return self.boxes2relpos(self.clone())

    @staticmethod
    def rotated_iou(boxes1, boxes2, ret_union=False):
        """Compute the IOU between the two set of rotated boxes in order

        Parameters
        ----------
        boxes1: aloscene.OrientedBoxes2D
            (n, 5)
        boxes2: aloscene.OrientedBoxes2D
            (n, 5)
        ret_union : bool, optional
            If True, return union area, by default False

        Returns
        -------
        Tensor or tuple of (Tensor, Tensor)
            IoU, shape (n,), rotated IOU between each pair of boxes\n
            Union area, shape (n,), union area between each pair of boxes (if `ret_union` True)
        """
        assert boxes1.shape[0] == boxes2.shape[0]
        if boxes1.shape[0] == 0:
            iou = torch.zeros((0,), device=boxes1.device)
            u = torch.zeros((0,), device=boxes1.device)
        else:
            # rotated iou code works with (B, N, 5)
            # so a virtual dimension is added for code compatibility
            corners1 = boxes1.corners()[None]  # (1, n, 5)
            corners2 = boxes2.corners()[None]  # (1, n, 5)
            boxes1 = boxes1.as_tensor()[None]  # (1, n, 4)
            boxes2 = boxes2.as_tensor()[None]  # (1, n, 4)
            inter_area, _ = oriented_box_intersection_2d(corners1, corners2)
            area1 = boxes1[:, :, 2] * boxes1[:, :, 3]
            area2 = boxes2[:, :, 2] * boxes2[:, :, 3]
            u = area1 + area2 - inter_area
            iou = inter_area / u
            # remove virtual dimension
            u = u[0]
            iou = iou[0]
        if ret_union:
            return iou, u
        else:
            return iou

    def rotated_iou_with(self, boxes2, ret_union=False):
        """Compute the IOU between the two set of rotated boxes in order

        Parameters
        ----------
        boxes2: aloscene.OrientedBoxes2D
            (n, 5)
        ret_union : bool, optional
            If True, return union area, by default False

        Returns
        -------
        Tensor or tuple of (Tensor, Tensor)
            IoU, shape (n,), rotated IOU between each pair of boxes\n
            Union area, shape (n,), union area between each pair of boxes (if `ret_union` True)
        """
        return self.rotated_iou(self, boxes2, ret_union=ret_union)

    @staticmethod
    def rotated_giou(boxes1, boxes2, enclosing_type: str = "smallest", ret_iou=False):
        """Calculate GIoU for 2 sets of rotated boxes in order

        Parameters
        ----------
        boxes1 : BoundingBoxes3D
            Shape (n, 7)
        boxes2 : BoundingBoxes3D
            Shape (n, 7)
        enclosing_type : str, optional
            Choose the algorithm for finding enclosing box :
                - aligned      # simple and naive. bad performance. fastest
                - pca          # approximated smallest box. slightly worse performance.
                - smallest     # [default]. brute force. smallest box. best performance. slowest
        ret_iou : bool, optional
            If True, return also rotated IoU , by default False

        Returns
        -------
        Tensor or tuple of (Tensor, Tensor)
            GIoU, of shape (n,)\n
            IoU, of shape (n,) (if `ret_iou3d` True)
        """
        assert boxes1.shape[0] == boxes2.shape[0]
        if boxes1.shape[0] == 0:
            giou = torch.zeros((0,), device=boxes1.device)
            iou = torch.zeros((0,), device=boxes1.device)
        else:
            # rotated iou code works with (B, N, 5)
            # so a virtual dimension is added for code compatibility
            boxes1 = boxes1.as_tensor()[None]
            boxes2 = boxes2.as_tensor()[None]
            giou, iou = cal_giou(boxes1, boxes2, enclosing_type)
            giou = giou[0]
            iou = iou[0]
        if ret_iou:
            return giou, iou
        return giou

    def rotated_giou_with(self, boxes2, enclosing_type="smallest", ret_iou=False):
        """Calculate GIoU for 2 sets of rotated boxes in order

        Parameters
        ----------
        boxes2 : BoundingBoxes3D
            Shape (n, 7)
        enclosing_type : str, optional
            Choose the algorithm for finding enclosing box :
                - aligned      # simple and naive. bad performance. fastest
                - pca          # approximated smallest box. slightly worse performance.
                - smallest     # [default]. brute force. smallest box. best performance. slowest
        ret_iou : bool, optional
            If True, return also rotated IoU , by default False

        Returns
        -------
        Tensor or tuple of (Tensor, Tensor)
            GIoU, of shape (n,)\n
            IoU, of shape (n,) (if `ret_iou3d` True)
        """
        return self.rotated_giou(self, boxes2, enclosing_type, ret_iou)


if __name__ == "__main__":
    # boxes1 = OrientedBoxes2D(
    #     torch.tensor([
    #         [0, 0, 2, 2, 0],
    #         [1, 1, 2, 3, np.pi/6],
    #         [1, 1, 1, 3, -np.pi/3],
    #         [3, 1, 4, 2, np.pi/4]
    #     ])
    # ).to(torch.device("cuda"))
    # boxes2 = OrientedBoxes2D(
    #     torch.tensor([
    #         [1, 1, 2, 2, 0],
    #         [5, 5, 2, 3, np.pi/6],
    #         [1, 1, 1, 3, -np.pi/3],
    #         [3, 1, 4, 2, np.pi/4]
    #     ])
    # ).to(torch.device("cuda"))
    # # boxes1.render()
    # iou = OrientedBoxes2D.rotated_iou(boxes1, boxes2)
    # giou = OrientedBoxes2D.rotated_giou(boxes1, boxes2)
    # print(iou)
    # print(giou)

    # boxes1 = OrientedBoxes2D(
    #     torch.tensor(
    #         [
    #             [0.5, 0.5, 0.2, 0.2, 0],
    #             [0.1, 0.1, 0.2, 0.3, np.pi / 6],
    #             [0.1, 0.8, 0.1, 0.3, -np.pi / 3],
    #             [0.6, 0.3, 0.4, 0.2, np.pi / 4],
    #         ],
    #         device=torch.device("cuda"),
    #     ),
    #     absolute=False,
    # )
    boxes1 = OrientedBoxes2D(
        torch.tensor(
            [[150.0000, 150.0000, 50.0000, 80.0000, -1.0472], [100.0000, 200.0000, 100.0000, 20.0000, 0.2618]]
        )
    )
    boxes2 = OrientedBoxes2D(
        torch.tensor(
            [[200.0000, 10.0000, 50.0000, 100.0000, np.pi / 2], [100.0000, 200.0000, 10.0000, 50.0000, 0.2618]]
        )
    )
    boxes1.get_view(size=(300, 300)).render()
