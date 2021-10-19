from __future__ import annotations
import torch
from torch import Tensor

from typing import Union
import numpy as np
import cv2

import aloscene
from aloscene.renderer import View
from aloscene.labels import Labels
from torchvision.ops.boxes import nms


class BoundingBoxes2D(aloscene.tensors.AugmentedTensor):
    """BoundingBoxes2D Augmented Tensor. Used to represents 2D boxes in space encoded as `xcyc` (xc, yc, width, height
    ), `yxyx` (y_min, x_min, y_max, x_max) or `xyxy` (x_min, y_min, x_max, y_max). The data must be
    at least 2 dimensional (N, None) where N is the number of boxes. The last dimension is supposed to be 4.

    If your're data is more than 2 dimensional you might need to set the `names` to ("B", "N", None) for batch
    dimension or ("T", "N", None) for the temporal dimension, or even ("B", "T", "N", None) for batch and temporal
    dimension.

    .. warning:: It is important to note that on `Frame`, BoundingBoxes2D are not mergeable by default\
        . Indeed, one `Frame` is likely to get more or less boxes than an other one. Therefore, if you want to create\
            BoundingBoxes2D with batch dimension, it is recommded to create BoundingBoxes2D tensors as part of a list \
            that enclose the batch dimension (same for the temporal dimension).

    >>> [BoundingBoxes2D(...), BoundingBoxes2D(...), BoundingBoxes2D(...)].

    Finally, batch & temporal dimension could also be stored like this:

    >>> [[BoundingBoxes2D(...), BoundingBoxes2D(...)], [BoundingBoxes2D(...), BoundingBoxes2D(...)]]



    Parameters
    ----------
    x: list | torch.Tensor | np.array
        BoundingBoxes2D data. See explanation above for details.
    boxes_format: str
        One of `xyxy` (y_min, x_min, y_max, x_max), `yxyx` (x_min, y_min, x_max, y_max) or `xcyc` (xc, yc, width,
        height)
    absolute: bool
        Whether your boxes are encoded as absolute value or relative values (between 0 and 1). If absolute is True,
        the `frane size` must be given.
    frame_size: tuple | None
        (Height & Width) of the relative frame.
    names: tuple
        Names of the dimensions : ("N", None) by default. See explanation above for more details.

    Notes
    -----
    Note on dimension:

    - C refers to the channel dimension
    - N refers to a dimension with a dynamic number of element.
    - H refers to the height of a `SpatialAugmentedTensor`
    - W refers to the width of a `SpatialAugmentedTensor`
    - B refers to the batch dimension
    - T refers to the temporal dimension

    Examples
    --------
    >>> boxes2d = aloscene.BoundingBoxes2D(
    ...     [[0.5, 0.5, 01, 01], [0.4, 0.30, 0.2, 0.3]],
    ...     points_format="xcyc", absolute=False
    ....)
    >>> boxes2d = aloscene.BoundingBoxes2D(
    ...     [[512, 458, 20, 30], [280, 200, 100, 42]],
    ...     points_format="xcyc", absolute=True, frame_size=(1200, 1200)
        )
    """

    FORMATS = ["xcyc", "xyxy", "yxyx"]

    @staticmethod
    def __new__(
        cls,
        x,
        boxes_format: str,
        absolute: bool,
        labels: Union[dict, Labels] = None,
        frame_size=None,
        names=("N", None),
        *args,
        **kwargs,
    ):
        tensor = super().__new__(cls, x, *args, names=names, **kwargs)

        # Add label
        tensor.add_child("labels", labels, align_dim=["N"], mergeable=True)

        if boxes_format not in BoundingBoxes2D.FORMATS:
            raise Exception(
                "BoundingBoxes2D:Format `{}` not supported. Cound be one of {}".format(
                    boxes_format, BoundingBoxes2D.FORMATS
                )
            )
        tensor.add_property("boxes_format", boxes_format)
        tensor.add_property("absolute", absolute)
        tensor.add_property("padded_size", None)

        if absolute and frame_size is None:
            raise Exception("If the boxes format are absolute, the `frame_size` must be set")
        assert frame_size is None or (isinstance(frame_size, tuple) and len(frame_size) == 2)
        tensor.add_property("frame_size", frame_size)

        return tensor

    def __init__(self, x, *args, **kwargs):
        super().__init__(x)

    def append_labels(self, labels: Labels, name: str = None):
        """Attach a set of labels to the boxes. The attached set of labels are supposed to be equal to the
        number of points. In other words, the N dimensions must match in both tensor.

        Parameters
        ----------
        labels: aloscene.Labels
            Set of labels to attached to the frame
        name: str
            If none, the label will be attached without name (if possible). Otherwise if no other unnamed
            labels are attached to the frame, the labels will be added to the set of labels.

        Examples
        --------
        >>> boxes2d = aloscene.BoundingBoxes2D([[0.5, 0.5, 0.1, 0.1], [0.2, 0.1, 0.05, 0.05]], "xcyc", False)
        >>> labels = aloscene.Labels([51, 24])
        >>> boxes2d.append_labels(labels)
        >>> boxes2d.labels
        >>>
        Or using named labels
        >>> boxes2d = aloscene.BoundingBoxes2D([[0.5, 0.5, 0.1, 0.1], [0.2, 0.1, 0.05, 0.05]], "xcyc", False)
        >>> labels_set_1 = aloscene.Labels([51, 24])
        >>> labels_set_2 = aloscene.Labels([51, 24])
        >>> boxes2d.append_labels(labels_set_1, "set1")
        >>> boxes2d.append_labels(labels_set_2, "set2")
        >>> boxes2d.labels["set1"]
        >>> boxes2d.labels["set2"]
        """
        self._append_child("labels", labels, name)

    def xcyc(self) -> BoundingBoxes2D:
        """Get a new BoundingBoxes2D Tensor with boxes following this format:
        [x_center, y_center, width, height]. Could be relative value (betwen 0 and 1)
        or absolute value based on the current Tensor representation.

        Examples
        --------
        >>> boxes_xcyc = boxes.xcyc()
        """
        tensor = self.clone()

        if tensor.boxes_format == "xcyc":
            return tensor
        elif tensor.boxes_format == "xyxy":
            # Convert from xyxy to xcyc
            labels = tensor.drop_children()
            xcyc_boxes = torch.cat(
                [tensor[:, :2] + ((tensor[:, 2:] - tensor[:, :2]) / 2), (tensor[:, 2:] - tensor[:, :2])], dim=1
            )
            xcyc_boxes.boxes_format = "xcyc"
            xcyc_boxes.set_children(labels)
            tensor.set_children(labels)
            return xcyc_boxes
        elif tensor.boxes_format == "yxyx":
            # Convert from yxyx to xcyc
            labels = tensor.drop_children()
            tensor = tensor.rename_(None)
            xcyc_boxes = torch.cat(
                [
                    tensor[:, :2].flip([1]) + ((tensor[:, 2:].flip([1]) - tensor[:, :2].flip([1])) / 2),
                    (tensor[:, 2:].flip([1]) - tensor[:, :2].flip([1])),
                ],
                dim=1,
            )
            tensor.reset_names()
            xcyc_boxes.reset_names()
            xcyc_boxes.boxes_format = "xcyc"
            xcyc_boxes.set_children(labels)
            tensor.set_children(labels)
            return xcyc_boxes
        else:
            raise Exception(f"BoundingBoxes2D:Do not know mapping from {tensor.boxes_format} to xcyc")

    def xyxy(self) -> BoundingBoxes2D:
        """Get a new BoundingBoxes2D Tensor with boxes following this format:
        [x1, y1, x2, y2]. Could be relative value (betwen 0 and 1)
        or absolute value based on the current Tensor representation.

        Examples
        --------
        >>> boxes_xyxy = boxes.xyxy()
        """
        tensor = self.clone()

        if tensor.boxes_format == "xcyc":
            labels = tensor.drop_children()
            # Convert from xcyc to xyxy
            n_tensor = torch.cat([tensor[:, :2] - (tensor[:, 2:] / 2), tensor[:, :2] + (tensor[:, 2:] / 2)], dim=1,)
            n_tensor.boxes_format = "xyxy"
            n_tensor.set_children(labels)
            return n_tensor
        elif tensor.boxes_format == "xyxy":
            return tensor
        elif tensor.boxes_format == "yxyx":
            labels = tensor.drop_children()
            tensor.rename_(None)
            # Convert from yxyx to xyxy
            n_tensor = torch.cat([tensor[:, :2].flip([1]), tensor[:, 2:].flip([1])], dim=1,)
            tensor.reset_names()
            n_tensor.reset_names()
            n_tensor.boxes_format = "xyxy"
            n_tensor.set_children(labels)
            tensor.set_children(labels)
            return n_tensor
        else:
            raise Exception(f"BoundingBoxes2D:Do not know mapping from {tensor.boxes_format} to xyxy")

    def yxyx(self) -> BoundingBoxes2D:
        """Get a new BoundingBoxes2D Tensor with boxes following this format:
        [y1, x1, y1, x1]. Could be relative value (betwen 0 and 1)
        or absolute value based on the current Tensor representation.

        Examples
        --------
        >>> boxes_yxyx = boxes.yxyx()
        """
        tensor = self.clone()

        if tensor.boxes_format == "xcyc":
            labels = tensor.drop_children()
            tensor.rename_(None)
            # Convert from xcyc to yxyx
            yxyx_boxes = torch.cat(
                [
                    tensor[:, :2].flip([1]) - (tensor[:, 2:].flip([1]) / 2),
                    tensor[:, :2].flip([1]) + (tensor[:, 2:].flip([1]) / 2),
                ],
                dim=1,
            )
            yxyx_boxes.reset_names()
            tensor.reset_names()
            yxyx_boxes.boxes_format = "yxyx"
            yxyx_boxes.set_children(labels)
            tensor.set_children(labels)
            return yxyx_boxes
        elif tensor.boxes_format == "xyxy":
            labels = tensor.drop_children()
            tensor.rename_(None)
            # Convert from xyxy to yxyx
            yxyx_boxes = torch.cat([tensor[:, :2].flip([1]), tensor[:, 2:].flip([1])], dim=1,)
            yxyx_boxes.reset_names()
            tensor.reset_names()
            yxyx_boxes.boxes_format = "yxyx"
            yxyx_boxes.set_children(labels)
            tensor.set_children(labels)
            return yxyx_boxes
        elif tensor.boxes_format == "yxyx":
            return tensor
        else:
            raise Exception(f"BoundingBoxes2D:Do not know mapping from {tensor.boxes_format} to yxyx")

    def abs_pos(self, frame_size) -> BoundingBoxes2D:
        """Get a new BoundingBoxes2D Tensor with absolute position
        relative to the given `frame_size`.

        Parameters
        ----------
        frame_size: tuple
            Used to know the frame size of the absolute boxes 2D.

        Examples
        --------
        >>> n_boxes = boxes.abs_pos((height, width))
        >>> # Or, if you have access to a `Frame` augmented tensor
        >>> n_boxes = boxes.abs_pos(frame.HW)
        """
        tensor = self.clone()

        # Back to relative before to get the absolute pos
        if tensor.absolute and frame_size != tensor.frame_size:

            if tensor.boxes_format == "xcyc" or tensor.boxes_format == "xyxy":
                tensor = tensor.div(
                    torch.tensor(
                        [tensor.frame_size[1], tensor.frame_size[0], tensor.frame_size[1], tensor.frame_size[0]],
                        device=tensor.device,
                    )
                )
            else:
                tensor = tensor.div(
                    torch.tensor(
                        [tensor.frame_size[0], tensor.frame_size[1], tensor.frame_size[0], tensor.frame_size[1]],
                        device=tensor.device,
                    )
                )

            if tensor.padded_size is not None:

                tensor.padded_size = (
                    (
                        tensor.padded_size[0][0] / tensor.frame_size[0],
                        tensor.padded_size[0][1] / tensor.frame_size[0],
                    ),
                    (
                        tensor.padded_size[1][0] / tensor.frame_size[1],
                        tensor.padded_size[1][1] / tensor.frame_size[1],
                    ),
                )
            tensor.absolute = False

        if not tensor.absolute:
            if tensor.boxes_format == "xcyc" or tensor.boxes_format == "xyxy":
                tensor = tensor.mul(
                    torch.tensor([frame_size[1], frame_size[0], frame_size[1], frame_size[0]], device=tensor.device)
                )
            else:
                tensor = tensor.mul(
                    torch.tensor([frame_size[0], frame_size[1], frame_size[0], frame_size[1]], device=tensor.device)
                )
            tensor.frame_size = frame_size
            tensor.absolute = True
            if tensor.padded_size is not None:

                tensor.padded_size = (
                    (
                        tensor.padded_size[0][0] * tensor.frame_size[0],
                        tensor.padded_size[0][1] * tensor.frame_size[0],
                    ),
                    (
                        tensor.padded_size[1][0] * tensor.frame_size[1],
                        tensor.padded_size[1][1] * tensor.frame_size[1],
                    ),
                )

        elif tensor.absolute and frame_size == tensor.frame_size:
            pass
        else:
            raise Exception("boxes2absposNot habndler error")

        return tensor

    def rel_pos(self) -> BoundingBoxes2D:
        """Get a new BoundingBoxes2D Tensor with absolute position
        relative to the given `frame_size`.

        Examples
        --------
        >>> n_boxes = boxes.rel_pos()
        """
        tensor = self.clone()

        if tensor.absolute:
            if tensor.boxes_format == "xcyc" or tensor.boxes_format == "xyxy":
                tensor = tensor.div(
                    torch.tensor(
                        [tensor.frame_size[1], tensor.frame_size[0], tensor.frame_size[1], tensor.frame_size[0]],
                        device=tensor.device,
                    )
                )
            else:
                tensor = tensor.div(
                    torch.tensor(
                        [tensor.frame_size[0], tensor.frame_size[1], tensor.frame_size[0], tensor.frame_size[1]],
                        device=tensor.device,
                    )
                )

            if tensor.padded_size is not None:

                tensor.padded_size = (
                    (
                        tensor.padded_size[0][0] / tensor.frame_size[0],
                        tensor.padded_size[0][1] / tensor.frame_size[0],
                    ),
                    (
                        tensor.padded_size[1][0] / tensor.frame_size[1],
                        tensor.padded_size[1][1] / tensor.frame_size[1],
                    ),
                )

        elif not tensor.absolute:
            pass
        tensor.absolute = False
        tensor.frame_size = None
        return tensor

    def get_with_format(self, boxes_format: str) -> BoundingBoxes2D:
        """Set boxes into the desired format (Inplace operation)

        Parameters
        ----------
        boxes_format: str
            One of `xyxy` (y_min, x_min, y_max, x_max), `yxyx` (x_min, y_min, x_max, y_max) or `xcyc` (xc, yc, width,
            height)

        Examples
        --------
        >>> boxes_xcyc = boxes.get_with_format("xcyc")
        >>> boxes_yxyx = boxes.get_with_format("yxyx")
        >>> boxes_xyxy = boxes.get_with_format("xyxy")
        """
        if boxes_format == "xcyc":
            return self.xcyc()
        elif boxes_format == "xyxy":
            return self.xyxy()
        elif boxes_format == "yxyx":
            return self.yxyx()
        else:
            raise Exception(f"desired boxes_format {boxes_format} is not handle")

    def _area(self, boxes):
        """Get the area of each box"""
        if boxes.boxes_format == "xcyc":
            boxes = boxes.as_tensor()
            return boxes[:, 2] * boxes[:, 3]
        elif boxes.boxes_format == "xyxy":
            boxes = boxes.as_tensor()
            return (boxes[:, 2] - boxes[:, 0]).mul(boxes[:, 3] - boxes[:, 1])
        elif boxes.boxes_format == "yxyx":
            boxes = boxes.as_tensor()
            return (boxes[:, 2] - boxes[:, 0]).mul(boxes[:, 3] - boxes[:, 1])
        else:
            raise Exception(f"desired boxes_format {boxes.boxes_format} is not handle to compute the area")

    def abs_area(self, frame_size: Union[tuple, None]) -> torch.Tensor:
        """Get the absolute area of the current boxes.

        Parameters
        ----------
        frame_size: tuple | None
            If the current boxes are already absolute, will simply compute the area based on the current frame_size.
            Otherwise, for relative boxes, one must give the reference `frame_size`.

        Examples
        --------
        >>> # With relative boxes
        >>> area = boxes.abs_area(frame.HW)
        >>> # With absolute boxes
        >>> area = boxes.abs_area()
        """
        if self.absolute:
            return self._area(self.clone())
        else:
            if frame_size is None:
                raise Exception("Boxes are encoded as relative, the frame size must be given to compute the area.")
            return self._area(self.abs_pos(frame_size))

    def rel_area(self) -> torch.Tensor:
        """Get the relative area of the current boxes.


        Examples
        --------
        >>> area = boxes.rel_area()
        """
        if self.absolute:
            return self._area(self.rel_pos())
        else:
            return self._area(self.clone())

    def area(self) -> torch.Tensor:
        """Get the current boxes area. The area
        will be relative to the frame size if the boxes are in a relative
        state. Otherwise, the area will be absolute.

        Examples
        --------
        >>> area = boxes.area()
        """
        if self.absolute:
            return self.abs_area(self.frame_size)
        else:
            return self.rel_area()

    _GLOBAL_COLOR_SET = np.random.uniform(0, 1, (300, 3))

    def get_view(self, frame: Tensor = None, size: tuple = None, labels_set: str = None, **kwargs):
        """Create a view of the boxes a frame

        Parameters
        ----------
        frame: aloscene.Frame
            Tensor of type Frame to display the boxes on. If the frameis None, a frame will be create on the fly.
        size: (tuple)
            (height, width) Desired size of the view. None by default
        labels_set: str
            If provided, the boxes will rely on this label set to display the boxes color. If labels_set
            is not provie while the boxes have multiple labels set, the boxes will be display with the same colors.
        """
        from aloscene import Frame

        if frame is not None:
            if len(frame.shape) > 3:
                raise Exception(f"Expect image of shape c,h,w. Found image with shape {frame.shape}")
            assert isinstance(frame, Frame)
        else:
            size = self.frame_size if self.absolute else (300, 300)
            frame = torch.zeros(3, int(size[0]), int(size[1]))
            frame = Frame(frame, names=("C", "H", "W"), normalization="01")

        if self.padded_size is not None:
            boxes_abs = self.fit_to_padded_size()
            boxes_abs = boxes_abs.xyxy().abs_pos(frame.HW)
        else:
            boxes_abs = self.xyxy().abs_pos(frame.HW)

        # Get an imave with values between 0 and 1
        frame = frame.norm01().cpu().rename(None).permute([1, 2, 0]).detach().contiguous().numpy()
        # Draw bouding boxes

        # Try to retrieve the associated label ID (if any)
        labels = boxes_abs.labels if isinstance(boxes_abs.labels, aloscene.Labels) else [None] * len(boxes_abs)
        if labels_set is not None and not isinstance(boxes_abs.labels, dict):
            raise Exception(
                f"Trying to display a boxes labels set ({labels_set}) while boxes do not have multiple set of labels"
            )
        elif labels_set is not None and isinstance(boxes_abs.labels, dict) and labels_set not in boxes_abs.labels:
            raise Exception(
                f"Trying to display a boxes labels set ({labels_set}) while boxes do not have this set. Avaiable set ("
                + f"{[key for key in boxes_abs.labels]}"
                + ") "
            )
        elif labels_set is not None:
            labels = boxes_abs.labels[labels_set]
            assert labels.encoding == "id"

        for box, label in zip(boxes_abs, labels):
            box = box.round()
            x1, y1, x2, y2 = box.as_tensor()
            color = (0, 1, 0)
            if label is not None:
                color = self._GLOBAL_COLOR_SET[int(label) % len(self._GLOBAL_COLOR_SET)]
                cv2.putText(
                    frame, str(int(label)), (int(x2), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
                )
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
        # Return the view to display
        return View(frame, **kwargs)

    def iou_with(self, boxes2, ret_union=False) -> torch.Tensor:
        """Compute the IOU between the two set of boxes

        Parameters
        ----------
        boxes2: aloscene.BoundingBoxes2D
            Set of boxes to compute the IOU with

        Examples
        --------
        >>> # Compute the IOU between each boxe of the current set and the `boxes2d` boxes.
        >>> iou = boxes.iou_with(boxes2)
        >>> # Compute the IOU between each pair of boxes of the current set
        >>> iou = boxes.iou_with(boxes)
        """
        boxes1 = self

        if boxes1.boxes_format != "xyxy":
            boxes1 = boxes1.xyxy()
        if boxes2.boxes_format != "xyxy":
            boxes2 = boxes2.xyxy()
        # TODO, should do a method that ensure two boxes are in the same format
        # and convert the boxes automaticly
        if boxes2.absolute != boxes1.absolute and boxes1.absolute:
            boxes2 = boxes2.abs_pos(boxes1.frame_size)
        elif boxes2.absolute != boxes1.absolute and not boxes1.absolute:
            boxes2 = boxes2.rel_pos()

        assert boxes1.boxes_format == boxes2.boxes_format
        assert boxes1.absolute == boxes2.absolute

        area1 = boxes1.area()
        area2 = boxes2.area()

        boxes1 = boxes1.as_tensor()
        boxes2 = boxes2.as_tensor()

        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        union = area1[:, None] + area2 - inter

        iou = inter / union

        if ret_union:
            return iou, union
        else:
            return iou

    def giou_with(self, boxes2) -> torch.Tensor:
        """
        Generalized IoU from https://giou.stanford.edu/
        The boxes should be in [x0, y0, x1, y1] format
        Returns a [N, M] pairwise matrix, where N = len(boxes1)
        and M = len(boxes2)

        Parameters
        ----------
        boxes2: aloscene.BoundingBoxes2D
            Set of boxes to compute the giou with

        Examples
        --------
        >>> # Compute the GIOU between each boxe of the current set and the `boxes2d` boxes.
        >>> giou = boxes.giou_with(boxes2)
        >>> # Compute the GIOU between each pair of boxes of the current set
        >>> giou = boxes.giou_with(boxes)
        """
        boxes1 = self
        if boxes1.boxes_format != "xyxy":
            boxes1 = boxes1.xyxy()
        if boxes2.boxes_format != "xyxy":
            boxes2 = boxes2.xyxy()

        if boxes2.absolute != boxes1.absolute and boxes1.absolute:
            boxes2 = boxes2.abs_pos()
        elif boxes2.absolute != boxes1.absolute and not boxes1.absolute:
            boxes2 = boxes2.rel_pos()

        assert boxes1.boxes_format == boxes2.boxes_format
        assert boxes1.absolute == boxes2.absolute

        # degenerate boxes gives inf / nan results
        # so do an early check
        try:
            assert (boxes1.as_tensor()[:, 2:] >= boxes1.as_tensor()[:, :2]).all(), f"{boxes1.as_tensor()}"
            assert (boxes2.as_tensor()[:, 2:] >= boxes2.as_tensor()[:, :2]).all(), f"{boxes2.as_tensor()}"
        except:
            print("boxes1", boxes1)
            print("boxes2", boxes2)
            assert (boxes1.as_tensor()[:, 2:] >= boxes1.as_tensor()[:, :2]).all(), f"{boxes1.as_tensor()}"
            assert (boxes2.as_tensor()[:, 2:] >= boxes2.as_tensor()[:, :2]).all(), f"{boxes2.as_tensor()}"

        iou, union = boxes1.iou_with(boxes2, ret_union=True)

        boxes1 = boxes1.as_tensor()
        boxes2 = boxes2.as_tensor()

        lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])  # left top corner
        rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])  # right bottom corner

        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        area = wh[:, :, 0] * wh[:, :, 1]

        return iou - (area - union) / area

    def nms(self, scores: torch.Tensor, iou_threshold: float = 0.5) -> torch.Tensor:
        """Perform NMS on the set of boxes. To be performed, the boxes one must passed
        a `scores` tensor.

        Parameters
        ----------
        scores: torch.Tensor
            Scores of each boxes to perform the NMS computation.
        iou_threshold: float
            NMS iou threshold

        Examples
        --------
        >>> # indices kept by the NMS
        >>> indices = boxes.nms(scores, iou_threshold=0.5)

        Returns
        -------
            int64 tensor
            The indices of the elements that have been kept by NMS, sorted in decreasing order of scores
        """
        nms_boxes = self.xyxy()
        return nms(nms_boxes.as_tensor(), scores, iou_threshold)

    def _hflip(self, **kwargs):
        """Flip boxes horizontally"""
        boxes = self.clone()
        absolute = boxes.absolute
        frame_size = boxes.frame_size
        boxes_format = boxes.boxes_format

        # Transform to relative position, set format
        boxes = boxes.rel_pos().xcyc()

        # Flip horizontally
        boxes = torch.tensor([1.0, 0.0, 0, 0]) - boxes
        boxes.mul_(torch.tensor([1.0, -1.0, -1.0, -1.0]))

        # Put back the instance into the same state as before
        if absolute:
            boxes = boxes.abs_pos(frame_size)
        boxes = boxes.get_with_format(boxes_format)

        return boxes

    def _resize(self, size, **kwargs):
        """Resize BoundingBoxes2D, but not their labels

        Parameters
        ----------
        size : tuple of float
            target size (H, W) in relative coordinates between 0 and 1

        Returns
        -------
        boxes : aloscene.BoundingBoxes2D
            resized boxes
        """
        boxes = self.clone()
        # no modification needed for relative coordinates
        if not boxes.absolute:
            return boxes
        else:
            abs_size = tuple(s * fs for s, fs in zip(size, boxes.frame_size))
            return boxes.abs_pos(abs_size)

    def _crop(self, H_crop: tuple, W_crop: tuple, **kwargs):
        """Crop Boxes with the given relative crop

        Parameters
        ----------
        H_crop: tuple
            (start, end) between 0 and 1
        W_crop: tuple
            (start, end) between 0 and 1

        Returns
        -------
        cropped_boxes2d sa_tensor: aloscene.BoundingBoxes2D
            cropped_boxes2d BoundingBoxes2D
        """
        if self.padded_size is not None:
            raise Exception("Can't crop when padded size is not Note. Call fit_to_padded_size() first")

        absolute = self.absolute
        frame_size = self.frame_size
        boxes_format = self.boxes_format

        # Get a new set of bbox
        n_boxes = self.abs_pos((100, 100)).xyxy()

        # Retrieve crop coordinates
        h = (H_crop[1] - H_crop[0]) * 100
        w = (W_crop[1] - W_crop[0]) * 100
        x, y = W_crop[0] * 100, H_crop[0] * 100

        # Crop boxes
        max_size = torch.as_tensor([w, h, w, h], dtype=torch.float32)
        cropped_boxes = n_boxes - torch.as_tensor([x, y, x, y])

        cropped_boxes = torch.min(cropped_boxes.rename(None), max_size).reset_names()
        cropped_boxes = cropped_boxes.clamp(min=0)

        # Filter to keep only boxes with area > 0
        area = cropped_boxes.area()
        cropped_boxes = cropped_boxes[area > 0]

        cropped_boxes.frame_size = (h, w)
        cropped_boxes = cropped_boxes.rel_pos()

        # Put back the instance into the same state as before
        if absolute:
            cropped_boxes = cropped_boxes.abs_pos(frame_size)

        cropped_boxes = cropped_boxes.get_with_format(boxes_format)

        return cropped_boxes

    def fit_to_padded_size(self):
        """This method can be usefull when one use a padded Frame but only want to learn on the non-padded area.
        Thefore the target boxes will remain unpadded while keeping information about the real padded size.

        Therefore. If the set of boxes did not get padded yet by the pad operation, this method wil pad the boxes to
        the real padded size.

        Examples
        --------
        >>> padded_boxes = boxes.fit_to_padded_size()
        """
        if self.padded_size is None:
            raise Exception("Trying to fit to padded size without any previous stored padded_size.")

        if not self.absolute:
            offset_y = (self.padded_size[0][0], self.padded_size[0][1])
            offset_x = (self.padded_size[1][0], self.padded_size[1][1])
        else:
            offset_y = (self.padded_size[0][0] / self.frame_size[0], self.padded_size[0][1] / self.frame_size[0])
            offset_x = (self.padded_size[1][0] / self.frame_size[1], self.padded_size[1][1] / self.frame_size[1])

        if not self.absolute:
            boxes = self.abs_pos((100, 100)).xcyc()
            h_shift = boxes.frame_size[0] * offset_y[0]
            w_shift = boxes.frame_size[1] * offset_x[0]
            boxes = boxes + torch.as_tensor([[w_shift, h_shift, 0, 0]], device=boxes.device)
            boxes.frame_size = (100 * (1.0 + offset_y[0] + offset_y[1]), 100 * (1.0 + offset_x[0] + offset_x[1]))
            boxes = boxes.get_with_format(self.boxes_format)
            boxes = boxes.rel_pos()
        else:
            boxes = self.xcyc()
            h_shift = boxes.frame_size[0] * offset_y[0]
            w_shift = boxes.frame_size[1] * offset_x[0]
            boxes = boxes + torch.as_tensor([[w_shift, h_shift, 0, 0]], device=boxes.device)
            boxes.frame_size = (
                boxes.frame_size[0] * (1.0 + offset_y[0] + offset_y[1]),
                boxes.frame_size[1] * (1.0 + offset_x[0] + offset_x[1]),
            )
            boxes = boxes.get_with_format(self.boxes_format)

        boxes.padded_size = None

        return boxes

    def _pad(self, offset_y: tuple, offset_x: tuple, pad_boxes: bool = True, **kwargs):
        """Pad the set of boxes based on the given offset

        Parameters
        ----------
        offset_y: tuple
            (percentage top_offset, percentage bottom_offset) Percentage based on the previous size
        offset_x: tuple
            (percentage left_offset, percentage right_offset) Percentage based on the previous size
        pad_boxes: bool
            By default, the boxes are not changed when we pad the frame. Therefore the boxes still
            encode the position of the boxes based on the frame before the padding. This is usefull in some
            cases, like in transformer architecture where the padded ares are masked. Therefore, the transformer
            do not "see" the padded part of the frames.

        Returns
        -------
        boxes2d: aloscene.BoundingBoxes2D
            padded_boxes2d BoundingBoxes2D or unchange BoundingBoxes2D (if pad_boxes is False)
        """

        if not pad_boxes:
            n_boxes = self.clone()



            if n_boxes.padded_size is not None:

                if n_boxes.absolute:
                    pr_frame_size = self.frame_size
                else:
                    pr_frame_size = (1, 1)

                n_padded_size = (
                    (
                        offset_y[0] * (pr_frame_size[0] + n_boxes.padded_size[0][0] + n_boxes.padded_size[0][1]),
                        offset_y[1] * (pr_frame_size[0] + n_boxes.padded_size[0][0] + n_boxes.padded_size[0][1]),
                    ),
                    (
                        offset_x[0] * (pr_frame_size[1] + n_boxes.padded_size[1][0] + n_boxes.padded_size[1][1]),
                        offset_x[1] * (pr_frame_size[1] + n_boxes.padded_size[1][0] + n_boxes.padded_size[1][1]),
                    ),
                )

                n_padded_size = (
                    (
                        n_boxes.padded_size[0][0] + n_padded_size[0][0],
                        n_boxes.padded_size[0][1] + n_padded_size[0][1],
                    ),
                    (
                        n_boxes.padded_size[1][0] + n_padded_size[1][0],
                        n_boxes.padded_size[1][1] + n_padded_size[1][1],
                    ),
                )
            else:
                n_padded_size = (
                    (offset_y[0], offset_y[1]),
                    (offset_x[0], offset_x[1]),
                )

            n_boxes.padded_size = n_padded_size

            return n_boxes

        if self.padded_size is not None:
            raise Exception(
                "Padding with pad_boxes True while padded_size is not None is not supported Yet. Call fit_to_padded_size() first."
            )

        if not self.absolute:
            boxes = self.abs_pos((100, 100)).xcyc()
            h_shift = boxes.frame_size[0] * offset_y[0]
            w_shift = boxes.frame_size[1] * offset_x[0]
            boxes = boxes + torch.as_tensor([[w_shift, h_shift, 0, 0]], device=boxes.device)
            boxes.frame_size = (100 * (1.0 + offset_y[0] + offset_y[1]), 100 * (1.0 + offset_x[0] + offset_x[1]))
            boxes = boxes.get_with_format(self.boxes_format)
            boxes = boxes.rel_pos()
        else:
            boxes = self.xcyc()
            h_shift = boxes.frame_size[0] * offset_y[0]
            w_shift = boxes.frame_size[1] * offset_x[0]
            boxes = boxes + torch.as_tensor([[w_shift, h_shift, 0, 0]], device=boxes.device)
            boxes.frame_size = (
                boxes.frame_size[0] * (1.0 + offset_y[0] + offset_y[1]),
                boxes.frame_size[1] * (1.0 + offset_x[0] + offset_x[1]),
            )
            boxes = boxes.get_with_format(self.boxes_format)

        return boxes

    def _spatial_shift(self, shift_y: float, shift_x: float, **kwargs):
        """
        Spatially shift the Boxes

        Parameters
        ----------
        shift_y: float
            Shift percentage on the y axis. Could be negative or positive
        shift_x: float
            Shift percentage on the x axis. Could ne negative or positive.

        Returns
        -------
        shifted_tensor: aloscene.AugmentedTensor
            shifted tensor
        """
        if self.padded_size is not None:
            raise Exception(
                "Can't process spatial shift when padded size is not Note. Call fit_to_padded_size() first"
            )

        original_format = self.boxes_format
        original_absolute = self.absolute
        frame_size = self.frame_size

        n_boxes = self.clone().rel_pos().xcyc()

        n_boxes += torch.as_tensor([[shift_x, shift_y, 0, 0]])  # , device=self.device)

        max_size = torch.as_tensor([1, 1, 1, 1], dtype=torch.float32)

        n_boxes = torch.min(n_boxes.rename(None), max_size)
        n_boxes = n_boxes.clamp(min=0)
        n_boxes = n_boxes.reset_names()
        # Filter to keep only boxes with area > 0
        area = n_boxes.area()
        n_boxes = n_boxes[area > 0]

        # Put back the instance into the same state as before
        if original_absolute:
            n_boxes = n_boxes.abs_pos(frame_size)
        n_boxes = n_boxes.get_with_format(original_format)

        return n_boxes

    def as_boxes(self, boxes: BoundingBoxes2D) -> BoundingBoxes2D:
        """Convert the current boxes state into the given `boxes` state, following the same `boxes_format`, the same
        `frame_size` (if any) and the same `padded_size` (if any).

        Parameters
        ----------
        boxes: aloscene.BoundingBoxes2D
            Boxes to match the format.

        Examples
        --------
        >>> n_boxes = boxes.as_boxes(other_boxes)
        """
        n_boxes = self.clone()

        if boxes.absolute and not n_boxes.absolute:
            n_boxes = n_boxes.abs_pos(boxes.frame_size)
        elif not boxes.absolute and n_boxes.absolute:
            n_boxes = n_boxes.rel_pos()

        n_boxes = n_boxes.get_with_format(boxes.boxes_format)

        if boxes.padded_size is not None:
            n_boxes.padded_size = boxes.padded_size

        return n_boxes

    def remove_padding(self) -> BoundingBoxes2D:
        """This method can be usefull when one use a padded Frame but only want to learn on the non-padded area.
        Thefore the target points will remain unpadded while keeping information about the real padded size.

        Thus, this method will simply remove the memorized padded information.

        Examples
        --------
        >>> n_boxes = boxes.remove_padding()
        """
        n_boxes = self.clone()
        n_boxes.padded_size = None
        return n_boxes
