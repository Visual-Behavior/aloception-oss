# from __future__ import annotations

from torchvision.io.image import read_image
import torch
from torch import Tensor
from torch._C import device
import torchvision

from typing import *
import numpy as np
import cv2

import aloscene
from aloscene.renderer import View
from aloscene.labels import Labels
import torchvision
from torchvision.ops.boxes import nms
from aloscene.renderer import View, put_adapative_cv2_text, adapt_text_size_to_frame


class Points2D(aloscene.tensors.AugmentedTensor):
    """Points2D Augmented Tensor. Used to represents 2D points in space encoded as xy or yx. The data must be
    at least 2 dimensional (N, None) where N is the number of points.

    If your're data is more than 2 dimensional you might need to set the `names` to ("B", "N", None) for batch
    dimension or ("T", "N", None) for the temporal dimension, or even ("B", "T", "N", None) for batch and temporal
    dimension.

    .. warning:: It is important to note that on `Frame`, Points2D are not mergeable by default\
        . Indeed, one `Frame` is likely to get more or less points than an other one. Therefore, if you want to create\
            Points2D with batch dimension, it is recommded to create Points2D tensors as part of a list that enclose \
                the batch dimension (same for the temporal dimension).

    >>> [Points2D(...), Points2D(...), Points2D(...)].

    Finally, batch & temporal dimension could also be stored like this:

    >>> [[Points2D(...), Points2D(...)], [Points2D(...), Points2D(...)]]



    Parameters
    ----------
    x: list | torch.Tensor | np.array
        Points2D data. See explanation above for details.
    points_format: str
        One of "xy", "yx". Whether your points are stored as "xy" or "yx".
    absolute: bool
        Whether your points are encoded as absolute value or relative values (between 0 and 1). If absolute is True,
        the `frane size` must be given.
    frame_size: tuple
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
    >>> pts2d = aloscene.Points2D(
    ...     [[0.5, 0.5], [0.4, 0.49]],
    ...     points_format="yx", absolute=False
    ....)
    >>> pts2d = aloscene.Points2D(
    ...     [[512, 458], [28, 20]],
    ...     points_format="yx", absolute=True, frame_size=(1200, 1200)
        )
    """

    FORMATS = ["xy", "yx"]

    @staticmethod
    def __new__(
        cls,
        x: Union[list, np.array, torch.Tensor],
        points_format: str,
        absolute: bool,
        labels: Union[dict, Labels, None] = None,
        frame_size=None,
        names=("N", None),
        *args,
        **kwargs,
    ):

        if points_format not in Points2D.FORMATS:
            raise Exception(
                "Point2d:Format `{}` not supported. Cound be one of {}".format(points_format, Points2D.FORMATS)
            )

        tensor = super().__new__(cls, x, *args, names=names, **kwargs)

        # Add label
        tensor.add_child("labels", labels, align_dim=["N"], mergeable=True)

        tensor.add_property("points_format", points_format)
        tensor.add_property("absolute", absolute)
        tensor.add_property("padded_size", None)

        if absolute and frame_size is None:
            raise Exception("If the points format are absolute, the `frame_size` must be set")
        assert frame_size is None or (isinstance(frame_size, tuple) and len(frame_size) == 2)
        tensor.add_property("frame_size", frame_size)

        return tensor

    def __init__(self, x, *args, **kwargs):
        super().__init__(x)

    def append_labels(self, labels: Labels, name: Union[str, None] = None):
        """Attach a set of labels to the points. The attached set of labels are supposed to be equal to the
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
        >>> pts2d = aloscene.Points2D([[0.5, 0.5], [0.2, 0.1]], "yx", False)
        >>> labels = aloscene.Labels([51, 24])
        >>> pts2d.append_labels(labels)
        >>> pts2d.labels
        >>>
        Or using named labels
        >>> pts2d = aloscene.Points2D([[0.5, 0.5], [0.2, 0.1]], "yx", False)
        >>> labels_set_1 = aloscene.Labels([51, 24])
        >>> labels_set_2 = aloscene.Labels([51, 24])
        >>> pts2d.append_labels(labels_set_1, "set1")
        >>> pts2d.append_labels(labels_set_2, "set2")
        >>> pts2d.labels["set1"]
        >>> pts2d.labels["set2"]
        """
        self._append_child("labels", labels, name)

    def xy(self):
        """Get a new Point2d Tensor with points following this format:
        [x, y]. Could be relative value (betwen 0 and 1)
        or absolute value based on the current Tensor representation.

        Examples
        --------
        >>> points_2d_xy = points.xy()
        """
        tensor = self.clone()
        if tensor.points_format == "xy":
            return tensor
        elif tensor.points_format == "yx":
            tensor = torch.cat([tensor[:, 1:2], tensor[:, 0:1]], dim=-1)
            tensor.points_format = "xy"
            return tensor

    def yx(self):
        """Get a new Point2d Tensor with points following this format:
        [y, x]. Could be relative value (betwen 0 and 1)
        or absolute value based on the current Tensor representation.

        Examples
        --------
        >>> points_2d_yx = points.yx()
        """
        tensor = self.clone()
        if tensor.points_format == "yx":
            return tensor
        elif tensor.points_format == "xy":
            tensor = torch.cat([tensor[:, 1:2], tensor[:, 0:1]], dim=-1)
            tensor.points_format = "yx"
            return tensor

    def abs_pos(self, frame_size: tuple):
        """Get a new Point2d Tensor with absolute position
        relative to the given `frame_size`.

        Parameters
        ----------
        frame_size: tuple
            Frame size associated with the absolute points2d. (height, width)

        Examples
        --------
        >>> points_2d_abs = points.abs_pos((height, width))
        """
        tensor = self.clone()

        # Back to relative before to get the absolute pos
        if tensor.absolute and frame_size != tensor.frame_size:

            if tensor.points_format == "xy":
                mul_tensor = torch.tensor([[frame_size[1], frame_size[0]]], device=self.device)
            else:
                mul_tensor = torch.tensor([[frame_size[0], frame_size[1]]], device=self.device)

            tensor = tensor.rel_pos()
            tensor = tensor.mul(mul_tensor)
            tensor.frame_size = frame_size
            tensor.absolute = True

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

            return tensor
        elif tensor.absolute and frame_size == tensor.frame_size:
            return tensor
        else:

            if tensor.points_format == "xy":
                mul_tensor = torch.tensor([[frame_size[1], frame_size[0]]], device=self.device)
            else:
                mul_tensor = torch.tensor([[frame_size[0], frame_size[1]]], device=self.device)

            tensor = tensor.mul(mul_tensor)
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

            return tensor

    def rel_pos(self):
        """Get a new Point2d Tensor with relative position (between 0 and 1)
        based on the current frame_size.

        Examples
        --------
        >>> points_2d_rel = points.rel_pos()
        """
        tensor = self.clone()

        # Back to relative before to get the absolute pos
        if tensor.absolute:
            if tensor.points_format == "xy":
                div_tensor = torch.tensor([[self.frame_size[1], self.frame_size[0]]], device=self.device)
            else:
                div_tensor = torch.tensor([[self.frame_size[0], self.frame_size[1]]], device=self.device)
            tensor = tensor.div(div_tensor)
            tensor.absolute = False

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

            return tensor
        else:
            return tensor

    def get_with_format(self, points_format: str):
        """Get the points with the desired format.

        Parameters
        ----------
        points_format: str
            One of ("xy", "yx")

        Examples
        --------
        >>> n_points = points.get_with_format("yx")
        """
        if points_format == "xy":
            return self.xy()
        elif points_format == "yx":
            return self.yx()
        else:
            raise Exception(f"desired points_format {points_format} is not handle")

    _GLOBAL_COLOR_SET = np.random.uniform(0, 1, (300, 3))

    def get_view(self, frame=None, size: Union[tuple, None] = None, labels_set: Union[str, None] = None, **kwargs):
        """Create a view of the points on a frame

        Parameters
        ----------
        frame: aloscene.Frame
            Tensor of type Frame to display the points on. If the frame is None, an empty frame will be create on the
            fly. If the frame is passed, the frame should be 3 dimensional ("C", "H", "W") or ("H", "W", "C")
        size: (tuple)
            (height, width) Desired size of the view. None by default
        labels_set: str
            If provided, the points will rely on this label set to display the points color. If labels_set
            is not provie while the points have multiple labels set, the points will be display with the same colors.

        Examples
        --------
        >>> points.get_view().render()
        >>> # Or using a frame to render the points
        >>> points.get_view(frame).render()
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
            points_abs = self.fit_to_padded_size()
            points_abs = points_abs.xy().abs_pos(frame.HW)
        else:
            points_abs = self.xy().abs_pos(frame.HW)

        # Get an imave with values between 0 and 1
        frame_size = (frame.H, frame.W)
        frame = frame.norm01().cpu().rename(None).permute([1, 2, 0]).detach().contiguous().numpy()
        # Draw points

        # Try to retrieve the associated label ID (if any)
        labels = points_abs.labels if isinstance(points_abs.labels, aloscene.Labels) else [None] * len(points_abs)
        if labels_set is not None and not isinstance(points_abs.labels, dict):
            raise Exception(
                f"Trying to display a set of points labels ({labels_set}) while the points do not have multiple set of labels"
            )
        elif labels_set is not None and isinstance(points_abs.labels, dict) and labels_set not in points_abs.labels:
            raise Exception(
                f"Trying to display a set of points labels ({labels_set}) while the points no not have this set. Avaiable set ("
                + [key for key in points_abs.labels]
                + ") "
            )
        elif labels_set is not None:
            labels = points_abs.labels[labels_set]
            assert labels.encoding == "id"

        size, _ = adapt_text_size_to_frame(1.0, frame_size)
        for box, label in zip(points_abs, labels):
            box = box.round()
            x1, y1 = box.as_tensor()
            color = (0, 1, 0)
            if label is not None:
                color = self._GLOBAL_COLOR_SET[int(label) % len(self._GLOBAL_COLOR_SET)]

                put_adapative_cv2_text(
                    frame,
                    frame_size,
                    str(int(label)),
                    pos_x=int(x1) + 10,
                    pos_y=int(y1) + 10,
                    color=color,
                    square_background=False,
                )

            cv2.circle(frame, (int(x1), int(y1)), int(size * 5), color, 2)
        # Return the view to display
        return View(frame, **kwargs)

    def _hflip(self, **kwargs):
        """Flip points horizontally"""
        points = self.clone()

        absolute = points.absolute
        frame_size = points.frame_size
        points_format = points.points_format

        # Transform to relative position, set format
        points = points.rel_pos().xy()

        # Flip horizontally
        points = torch.tensor([1.0, 0.0]) - points
        points.mul_(torch.tensor([1.0, -1.0]))

        # Put back the instance into the same state as before
        if absolute:
            points = points.abs_pos(frame_size)
        points = points.get_with_format(points_format)

        return points

    def _resize(self, size, **kwargs):
        """Resize Point2d, but not their labels

        Parameters
        ----------
        size : tuple of float
            target size (H, W) in relative coordinates between 0 and 1

        Returns
        -------
        points : aloscene.Point2d
            resized points
        """
        points = self.clone()
        # no modification needed for relative coordinates
        if not points.absolute:
            return points
        else:
            abs_size = tuple(s * fs for s, fs in zip(size, points.frame_size))
            return points.abs_pos(abs_size)

    def _rotate(self, angle, **kwargs):
        """Rotate Point2d, but not their labels

        Parameters
        ----------
        size : float

        Returns
        -------
        points : aloscene.Point2d
            rotated points
        """
        points = self.xy()
        H, W = self.frame_size
        angle_rad = angle * np.pi / 180
        rot_mat = torch.tensor([[np.cos(angle_rad), np.sin(angle_rad)], [-np.sin(angle_rad), np.cos(angle_rad)]]).to(
            torch.float32
        )
        tr_mat = torch.tensor([W / 2, H / 2])
        for i in range(points.shape[0]):
            points[i] = torch.matmul(rot_mat, points[i] - tr_mat) + tr_mat

        max_size = torch.as_tensor([W, H], dtype=torch.float32)
        points_filter = (points >= 0).as_tensor() & (points <= max_size).as_tensor()
        points_filter = points_filter[:, 0] & points_filter[:, 1]
        points = points[points_filter]

        points = points.rel_pos()
        # no modification needed for relative coordinates
        if self.absolute:
            points = points.abs_pos(self.frame_size)

        points = points.get_with_format(self.points_format)

        return points

    def _crop(self, H_crop: tuple, W_crop: tuple, **kwargs):
        """Crop Points with the given relative crop

        Parameters
        ----------
        H_crop: tuple
            (start, end) between 0 and 1
        W_crop: tuple
            (start, end) between 0 and 1
        """
        if self.padded_size is not None:
            raise Exception("Can't crop when padded size is not Note. Call fit_to_padded_size() first")

        absolute = self.absolute
        frame_size = self.frame_size

        points_format = self.points_format

        # Get a new set of points
        n_points = self.abs_pos((100, 100)).xy()

        # Retrieve crop coordinates
        h = (H_crop[1] - H_crop[0]) * 100
        w = (W_crop[1] - W_crop[0]) * 100
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        x, y = W_crop[0] * 100, H_crop[0] * 100

        # Crop points
        cropped_points = n_points - torch.as_tensor([x, y])

        cropped_points_filter = (cropped_points >= 0).as_tensor() & (cropped_points < max_size).as_tensor()
        cropped_points_filter = cropped_points_filter[:, 0] & cropped_points_filter[:, 1]
        cropped_points = cropped_points[cropped_points_filter]

        cropped_points.frame_size = (h, w)
        cropped_points = cropped_points.rel_pos()

        # Put back the instance into the same state as before
        if absolute:
            n_frame_size = ((H_crop[1] - H_crop[0]) * frame_size[0], (W_crop[1] - W_crop[0]) * frame_size[1])
            cropped_points = cropped_points.abs_pos(n_frame_size)
        else:
            cropped_points.frame_size = None

        cropped_points = cropped_points.get_with_format(points_format)

        return cropped_points

    def fit_to_padded_size(self):
        """This method can be usefull when one use a padded Frame but only want to learn on the non-padded area.
        Thefore the target points will remain unpadded while keeping information about the real padded size.

        Therefore. If the set of points did not get padded yet by the pad operation, this method wil pad the points to
        the real padded size.

        Examples
        --------
        >>> padded_points = points.fit_to_padded_size()
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
            points = self.abs_pos((100, 100)).xy()
            h_shift = points.frame_size[0] * offset_y[0]
            w_shift = points.frame_size[1] * offset_x[0]
            points = points + torch.as_tensor([[w_shift, h_shift]], device=points.device)
            points.frame_size = (100 * (1.0 + offset_y[0] + offset_y[1]), 100 * (1.0 + offset_x[0] + offset_x[1]))
            points = points.get_with_format(self.points_format)
            points = points.rel_pos()
        else:
            points = self.xy()
            h_shift = points.frame_size[0] * offset_y[0]
            w_shift = points.frame_size[1] * offset_x[0]
            points = points + torch.as_tensor([[w_shift, h_shift]], device=points.device)
            points.frame_size = (
                points.frame_size[0] * (1.0 + offset_y[0] + offset_y[1]),
                points.frame_size[1] * (1.0 + offset_x[0] + offset_x[1]),
            )
            points = points.get_with_format(self.points_format)

        points.padded_size = None

        return points

    def _pad(self, offset_y: tuple, offset_x: tuple, pad_points2d: bool = True, **kwargs):
        """Pad the set of points based on the given offset

        Parameters
        ----------
        offset_y: tuple
            (percentage top_offset, percentage bottom_offset) Percentage based on the previous size
        offset_x: tuple
            (percentage left_offset, percentage right_offset) Percentage based on the previous size
        pad_points: bool
            By default, the points are not changed when we pad the frame. Therefore the points still
            encode the position of the points based on the frame before the padding. This is usefull in some
            cases, like in transformer architecture where the padded ares are masked. Therefore, the transformer
            do not "see" the padded part of the frames.
        """

        if not pad_points2d:

            n_points = self.clone()

            if n_points.absolute:
                pr_frame_size = self.frame_size
            else:
                pr_frame_size = (1, 1)

            n_padded_size = (
                (pr_frame_size[0] * offset_y[0], pr_frame_size[0] * offset_y[1]),
                (pr_frame_size[1] * offset_x[0], pr_frame_size[1] * offset_x[1]),
            )

            if n_points.padded_size is not None:
                raise Exception(
                    "Padding twice using pad_points False is not supported Yet. Call fit_to_padded_size() first."
                )

            n_points.padded_size = n_padded_size

            return n_points

        if self.padded_size is not None:
            raise Exception("Padding with pad_points True while padded_size is not None is not supported Yet.")

        if not self.absolute:
            points = self.abs_pos((100, 100)).xy()
            h_shift = points.frame_size[0] * offset_y[0]
            w_shift = points.frame_size[1] * offset_x[0]
            points = points + torch.as_tensor([[w_shift, h_shift]], device=points.device)
            points.frame_size = (100 * (1.0 + offset_y[0] + offset_y[1]), 100 * (1.0 + offset_x[0] + offset_x[1]))
            points = points.get_with_format(self.points_format)
            points = points.rel_pos()
        else:
            points = self.xy()
            h_shift = points.frame_size[0] * offset_y[0]
            w_shift = points.frame_size[1] * offset_x[0]
            points = points + torch.as_tensor([[w_shift, h_shift]], device=points.device)
            points.frame_size = (
                points.frame_size[0] * (1.0 + offset_y[0] + offset_y[1]),
                points.frame_size[1] * (1.0 + offset_x[0] + offset_x[1]),
            )
            points = points.get_with_format(self.points_format)

        return points

    def _spatial_shift(self, shift_y: float, shift_x: float, **kwargs):
        """
        Spatially shift the Points
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
        # get needed information
        original_format = self.points_format
        original_absolute = self.absolute
        frame_size = self.frame_size

        # shift the points
        n_points = self.clone().rel_pos().xy()
        n_points += torch.as_tensor([[shift_x, shift_y]])  # , device=self.device)

        # filter points outside of the image
        points_filter = (n_points >= 0).as_tensor() & (n_points <= 1).as_tensor()
        points_filter = points_filter[:, 0] & points_filter[:, 1]
        n_points = n_points[points_filter]
        n_points = n_points.reset_names()

        # Put back the instance into the same state as before
        if original_absolute:
            n_points = n_points.abs_pos(frame_size)
        n_points = n_points.get_with_format(original_format)

        return n_points

    def as_points(self, points):
        n_points = self.clone()

        if points.absolute and not n_points.absolute:
            n_points = n_points.abs_pos(points.frame_size)
        elif not points.absolute and n_points.absolute:
            n_points = n_points.rel_pos()

        n_points = n_points.get_with_format(points.points_format)

        if points.padded_size is not None:
            n_points.padded_size = points.padded_size

        return n_points

    def remove_padding(self):
        """This method can be usefull when one use a padded Frame but only want to learn on the non-padded area.
        Thefore the target points will remain unpadded while keeping information about the real padded size.

        Thus, this method will simply remove the memorized padded information.

        Returns:
            [type]: [description]
        """
        n_points = self.clone()
        n_points.padded_size = None
        return n_points
