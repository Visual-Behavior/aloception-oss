from __future__ import annotations
import torch
from torch import Tensor
import torchvision

from typing import *
import numpy as np
import cv2

import aloscene
from aloscene.renderer import View
from aloscene.labels import Labels
import torchvision
from torchvision.ops.boxes import nms


class BoundingBoxes2D(aloscene.tensors.AugmentedTensor):
    """Boxes2D Tensor."""

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
        tensor.add_label("labels", labels, align_dim=["N"], mergeable=True)

        if boxes_format not in BoundingBoxes2D.FORMATS:
            raise Exception(
                "BoundingBoxes2D:Format `{}` not supported. Cound be one of {}".format(
                    tensor.boxes_format, BoundingBoxes2D.FORMATS
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
        """Attach a set of labels to the boxes.

        Parameters
        ----------
        labels: aloscene.Labels
            Set of labels to attached to the frame
        name: str
            If none, the label will be attached without name (if possible). Otherwise if no other unnamed
            labels are attached to the frame, the labels will be added to the set of labels.
        """
        self._append_label("labels", labels, name)

    @staticmethod
    def boxes2xcyc(tensor):
        """Get a new BoundingBoxes2D Tensor with boxes following this format:
        [x_center, y_center, width, height]. Could be relative value (betwen 0 and 1)
        or absolute value based on the current Tensor representation.
        """
        if tensor.boxes_format == "xcyc":
            return tensor
        elif tensor.boxes_format == "xyxy":
            # Convert from xyxy to xcyc
            labels = tensor.drop_labels()
            xcyc_boxes = torch.cat(
                [tensor[:, :2] + ((tensor[:, 2:] - tensor[:, :2]) / 2), (tensor[:, 2:] - tensor[:, :2])], dim=1
            )
            xcyc_boxes.boxes_format = "xcyc"
            xcyc_boxes.set_labels(labels)
            tensor.set_labels(labels)
            return xcyc_boxes
        elif tensor.boxes_format == "yxyx":
            # Convert from yxyx to xcyc
            labels = tensor.drop_labels()
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
            xcyc_boxes.set_labels(labels)
            tensor.set_labels(labels)
            return xcyc_boxes
        else:
            raise Exception(f"BoundingBoxes2D:Do not know mapping from {tensor.boxes_format} to xcyc")

    def xcyc(self):
        """Get a new BoundingBoxes2D Tensor with boxes following this format:
        [x_center, y_center, width, height]. Could be relative value (betwen 0 and 1)
        or absolute value based on the current Tensor representation.
        """
        return self.boxes2xcyc(self.clone())

    @staticmethod
    def boxes2xyxy(tensor):
        if tensor.boxes_format == "xcyc":
            labels = tensor.drop_labels()
            # Convert from xcyc to xyxy
            n_tensor = torch.cat(
                [
                    tensor[:, :2] - (tensor[:, 2:] / 2),
                    tensor[:, :2] + (tensor[:, 2:] / 2),
                ],
                dim=1,
            )
            n_tensor.boxes_format = "xyxy"
            n_tensor.set_labels(labels)
            return n_tensor
        elif tensor.boxes_format == "xyxy":
            return tensor
        elif tensor.boxes_format == "yxyx":
            labels = tensor.drop_labels()
            tensor.rename_(None)
            # Convert from yxyx to xyxy
            n_tensor = torch.cat(
                [
                    tensor[:, :2].flip([1]),
                    tensor[:, 2:].flip([1]),
                ],
                dim=1,
            )
            tensor.reset_names()
            n_tensor.reset_names()
            n_tensor.boxes_format = "xyxy"
            n_tensor.set_labels(labels)
            tensor.set_labels(labels)
            return n_tensor
        else:
            raise Exception(f"BoundingBoxes2D:Do not know mapping from {tensor.boxes_format} to xyxy")

    def xyxy(self):
        """Get a new BoundingBoxes2D Tensor with boxes following this format:
        [x1, y1, x2, y2]. Could be relative value (betwen 0 and 1)
        or absolute value based on the current Tensor representation.
        """
        return self.boxes2xyxy(self.clone())

    @staticmethod
    def boxes2yxyx(tensor):
        if tensor.boxes_format == "xcyc":
            labels = tensor.drop_labels()
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
            yxyx_boxes.set_labels(labels)
            tensor.set_labels(labels)
            return yxyx_boxes
        elif tensor.boxes_format == "xyxy":
            labels = tensor.drop_labels()
            tensor.rename_(None)
            # Convert from xyxy to yxyx
            yxyx_boxes = torch.cat(
                [
                    tensor[:, :2].flip([1]),
                    tensor[:, 2:].flip([1]),
                ],
                dim=1,
            )
            yxyx_boxes.reset_names()
            tensor.reset_names()
            yxyx_boxes.boxes_format = "yxyx"
            yxyx_boxes.set_labels(labels)
            tensor.set_labels(labels)
            return yxyx_boxes
        elif tensor.boxes_format == "yxyx":
            return tensor
        else:
            raise Exception(f"BoundingBoxes2D:Do not know mapping from {tensor.boxes_format} to yxyx")

    def yxyx(self):
        """Get a new BoundingBoxes2D Tensor with boxes following this format:
        [y1, x1, y1, x1]. Could be relative value (betwen 0 and 1)
        or absolute value based on the current Tensor representation.
        """
        return self.boxes2yxyx(self.clone())

    @staticmethod
    def boxes2abspos(tensor, frame_size):
        """Get a new BoundingBoxes2D Tensor with absolute position
        relative to the given `frame_size`.
        """
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
            tensor.absolute = False
            if tensor.padded_size is not None:
                tensor.padded_size = (
                    tensor.padded_size[0] / tensor.frame_size[0] * 100,
                    tensor.padded_size[1] / tensor.frame_size[1] * 100,
                )

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
                    tensor.frame_size[0] * tensor.padded_size[0] / 100,
                    tensor.frame_size[1] * tensor.padded_size[1] / 100,
                )

        elif tensor.absolute and frame_size == tensor.frame_size:
            pass
        else:
            raise Exception("boxes2absposNot habndler error")

        return tensor

    def abs_pos(self, frame_size) -> BoundingBoxes2D:
        """Get a new BoundingBoxes2D Tensor with absolute position
        relative to the given `frame_size`.
        """
        return self.boxes2abspos(self.clone(), frame_size)

    @staticmethod
    def boxes2relpos(tensor):
        """Get a new BoundingBoxes2D Tensor with relative position
        (between 0 and 1)
        """
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
                    tensor.padded_size[0] / tensor.frame_size[0] * 100,
                    tensor.padded_size[1] / tensor.frame_size[1] * 100,
                )

        elif not tensor.absolute:
            pass
        tensor.absolute = False
        tensor.frame_size = None
        return tensor

    def rel_pos(self):
        """Get a new BoundingBoxes2D Tensor with absolute position
        relative to the given `frame_size`.
        """
        return self.boxes2relpos(self.clone())

    def get_with_format(self, boxes_format):
        """Set boxes into the desired format (Inplace operation)"""
        if boxes_format == "xcyc":
            return self.xcyc()
        elif boxes_format == "xyxy":
            return self.xyxy()
        elif boxes_format == "yxyx":
            return self.yxyx()
        else:
            raise Exception(f"desired boxes_format {boxes_format} is not handle")

    @staticmethod
    def boxes_hflip(boxes):
        """Flip frame horizontally"""
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

    def abs_area(self, size):
        """Get the absolute area"""
        if self.absolute:
            return self._area(self.clone())
        else:
            return self._area(self.abs_pos(size))

    def rel_area(self):
        """Get the absolute area"""
        if self.absolute:
            return self._area(self.rel_pos())
        else:
            return self._area(self.clone())

    def area(self):
        """Get the current boxes area. The area
        will be relative to the frame size if the boxes are in a relative
        state. Otherwise, the area will be absolute.
        """
        if self.absolute:
            return self.abs_area(self.frame_size)
        else:
            return self.rel_area()

    GLOBAL_COLOR_SET = np.random.uniform(0, 1, (300, 3))

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
                f"Trying to display a set of boxes labels ({labels_set}) while the boxes do not have multiple set of labels"
            )
        elif labels_set is not None and isinstance(boxes_abs.labels, dict) and labels_set not in boxes_abs.labels:
            raise Exception(
                f"Trying to display a set of boxes labels ({labels_set}) while the boxes no not have this set. Avaiable set ("
                + [key for key in boxes_abs.labels]
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
                color = self.GLOBAL_COLOR_SET[int(label) % len(self.GLOBAL_COLOR_SET)]
                cv2.putText(
                    frame, str(int(label)), (int(x2), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
                )
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
        # Return the view to display
        return View(frame, **kwargs)

    @staticmethod
    def iou(boxes1, boxes2, ret_union=False):
        """Compute the IOU between the two set of boxes

        Parameters
        ----------
        boxes1: aloscene.BoundingBoxes2D
        boxes2: aloscene.BoundingBoxes2D

        Returns
        -------
        iou_tensor: torch.Tensor
            IOU between each boxes
        """
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

    def iou_with(self, boxes2, ret_union=False):
        """Compute the IOU between the two set of boxes

        Parameters
        ----------
        boxes2: aloscene.BoundingBoxes2D

        Returns
        -------
        iou_tensor: torch.Tensor
            IOU between each boxes
        """
        return self.iou(self, boxes2, ret_union=ret_union)

    @staticmethod
    def giou(boxes1, boxes2):
        """
        Generalized IoU from https://giou.stanford.edu/
        The boxes should be in [x0, y0, x1, y1] format
        Returns a [N, M] pairwise matrix, where N = len(boxes1)
        and M = len(boxes2)

        Parameters
        ----------
        boxes1: aloscene.BoundingBoxes2D
        boxes2: aloscene.BoundingBoxes2D

        Returns
        -------
        giou_tensor: torch.Tensor
            Giou between each boxes
        """
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

    def giou_with(self, boxes2):
        """
        Generalized IoU from https://giou.stanford.edu/
        The boxes should be in [x0, y0, x1, y1] format
        Returns a [N, M] pairwise matrix, where N = len(boxes1)
        and M = len(boxes2)

        Parameters
        ----------
        boxes2: aloscene.BoundingBoxes2D

        Returns
        -------
        giou_tensor: torch.Tensor
            Giou between each boxes
        """
        return self.giou(self, boxes2)

    def nms(self, scores: torch.Tensor, iou_threshold: float = 0.5):
        """Perform NMS on the set of boxes. To be performed, the boxes one must passed
        a `scores` tensor.

        Parameters
        ----------
        scores: torch.Tensor
            Scores of each boxes to perform the NMS computation.
        iou_threshold: float
            NMS iou threshold

        Returns
        -------
            int64 tensor with the indices of the elements that have been kept by NMS, sorted in decreasing order of scores
        """
        nms_boxes = self.xyxy()

        return nms(nms_boxes.as_tensor(), scores, iou_threshold)

    def _hflip(self, **kwargs):
        """Flip boxes horizontally"""
        return self.boxes_hflip(self.clone())

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
        """If the set of Boxes did not get padded by the pad operation,
        this method wil padd the boxes to the real padded size.

        Returns
        -------
        padded_boxes2d sa_tensor: aloscene.BoundingBoxes2D
            padded_boxes2d BoundingBoxes2D
        """
        if self.padded_size is None:
            raise Exception("Trying to fit to padded size without any previous stored padded_size.")

        if not self.absolute:
            frame_size = (100, 100)  # Virtual frame size
        else:
            frame_size = self.frame_size

        offset_x = (0, self.padded_size[1] / frame_size[1])
        offset_y = (0, self.padded_size[0] / frame_size[0])

        if not self.absolute:
            boxes = self.abs_pos((100, 100))
            boxes.frame_size = (100 * offset_y[1], 100 * offset_x[1])
            boxes = boxes.rel_pos()
        else:
            boxes = self.clone()
            boxes.frame_size = (round(boxes.frame_size[0] * (offset_y[1])), round(boxes.frame_size[1] * (offset_x[1])))

        boxes.padded_size = None

        return boxes

    def _pad(self, offset_y: tuple, offset_x: tuple, pad_boxes: bool = False, **kwargs):
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
        assert offset_y[0] == 0 and offset_x[0] == 0, "Not handle yet"

        if not pad_boxes:

            n_boxes = self.clone()

            if n_boxes.padded_size is not None:
                pr_frame_size = n_boxes.padded_size
            elif n_boxes.padded_size is None and n_boxes.absolute:
                pr_frame_size = self.frame_size
            else:
                pr_frame_size = (100, 100)

            n_boxes.padded_size = (pr_frame_size[0] * (1.0 + offset_y[1]), pr_frame_size[1] * (1.0 + offset_x[1]))
            return n_boxes

        if self.padded_size is not None:
            raise Exception("Padding with pad_boxes True while padded_size is None is not supported Yet.")

        if not self.absolute:
            boxes = self.abs_pos((100, 100))
            boxes.frame_size = (100 * (1.0 + offset_y[1]), 100 * (1.0 + offset_x[1]))
            boxes = boxes.rel_pos()
        else:
            boxes = self.clone()
            boxes.frame_size = (boxes.frame_size[0] * (offset_y[1] + 1.0), boxes.frame_size[1] * (offset_x[1] + 1.0))

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

    def as_boxes(self, boxes):
        n_boxes = self.clone()

        if boxes.absolute and not n_boxes.absolute:
            n_boxes = n_boxes.abs_pos(boxes.frame_size)
        elif not boxes.absolute and n_boxes.absolute:
            n_boxes = n_boxes.rel_pos()

        n_boxes = n_boxes.get_with_format(boxes.boxes_format)

        if boxes.padded_size is not None:
            n_boxes.padded_size = boxes.padded_size

        return n_boxes

    def remove_padding(self):
        n_boxes = self.clone()
        n_boxes.padded_size = None
        return n_boxes
