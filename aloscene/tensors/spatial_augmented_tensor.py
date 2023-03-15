import matplotlib.pyplot as plt
import math
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
import torch
from typing import Union

from renderer import View, Renderer
from .augmented_tensor import AugmentedTensor
import inspect
import aloscene
from aloscene.camera_calib import CameraExtrinsic, CameraIntrinsic
from aloscene.utils.data_utils import LDtoDL

import warnings

class SpatialAugmentedTensor(AugmentedTensor):
    """Spatial Augmented Tensor. Used to represets any 2D data. The spatial augmented tensor can be used as a
    basis for images, depth or and spatially related data. Moreover, for stereo setup, the augmented tensor
    can encode data about the baseline and/or the side of the current plane ("left" or "right").

    cam_intrinsic: CameraIntrinsic
        Camera Intrinsic. If provided, the focal_length, plane_size, principal_point, camera_side & baseline will not
        be used. An error will be raised if theses parameters are provided since using the `cam_intrinsic` and theses
        parameters could be ambigious.
    cam_extrinsic: CameraExtrinsic
        Camera extrinsic parameters as an homogenious transformation matrix.
    camera_side : {'left', 'right', None}
        If part of a stereo setup, will encode the side of the camere "left" or "right" or None.
    baseline: float | None
        If part of a stereo setup, the `baseline` is the distance between the two cameras.
    timestamp: float | None
        Time in seconds since the beginning of the sequence.
    """

    @staticmethod
    def __new__(
        cls,
        x,
        *args,
        cam_intrinsic: Union[CameraIntrinsic, None] = None,
        cam_extrinsic: Union[CameraExtrinsic, None] = None,
        # For stereo setups
        camera_side: Union[str, None] = None,
        baseline: Union[float, None] = None,
        mask=None,
        projection="pinhole",
        distortion=1.0,
        **kwargs,
    ):
        tensor = super().__new__(cls, x, *args, **kwargs)

        tensor.add_child("mask", mask, align_dim=["B", "T"], mergeable=True)
        tensor.add_property("baseline", baseline)
        tensor.add_property("camera_side", camera_side)
        tensor.add_property("projection", projection)
        tensor.add_property("distortion", distortion)
        tensor.add_property("timestamp", None)

        # Intrisic and extrinsic parameters are cloned by default, to prevent having multiple reference
        # of the same intrisic/extrinsic across nodes.
        cam_intrinsic = cam_intrinsic.clone() if cam_intrinsic is not None else cam_intrinsic
        cam_extrinsic = cam_extrinsic.clone() if cam_extrinsic is not None else cam_extrinsic
        tensor.add_child("cam_intrinsic", cam_intrinsic, align_dim=["B", "T"], mergeable=True)
        tensor.add_child("cam_extrinsic", cam_extrinsic, align_dim=["B", "T"], mergeable=True)

        return tensor

    def __init__(self, x, *args, **kwargs):
        super().__init__(x)

    @classmethod
    def dummy(cls, size: tuple, names: tuple):
        dummy_class = cls(torch.ones(size))
        mask_size = list(size)
        if "C" in names:
            mask_size[names.index("C")] = 1
        mask_size = tuple(mask_size)
        dummy_class.append_mask(aloscene.Mask(torch.ones(mask_size), names=names))
        return dummy_class

    @property
    def HW(self):
        return (self.H, self.W)

    @property
    def W(self):
        return self.shape[self.names.index("W")]

    @property
    def H(self):
        return self.shape[self.names.index("H")]

    def append_mask(self, mask, name: Union[str, None] = None):
        """Attach a mask to the frame.
        The value 1 mean invalid and 0 mean valid.

        Parameters
        ----------
        mask: aloscene.Mask
            Mask to attached to the Frame
        name: str
            If none, the mask will be attached without name (if possible). Otherwise if no other unnamed
            mask are attached to the frame, the mask will be added to the set of mask.
        """
        self._append_child("mask", mask, name)

    def append_cam_intrinsic(self, cam_intrinsic: CameraIntrinsic):
        self._append_child("cam_intrinsic", cam_intrinsic)

    def append_cam_extrinsic(self, cam_extrinsic: CameraExtrinsic):
        self._append_child("cam_extrinsic", cam_extrinsic)

    def get_view(self, views: list = [], exclude=[], size=None, grid_size=None, title=None, add_title=True, **kwargs):
        """Render the spatial augmented tensor.

        Parameters
        ----------
        views: list
            List of `View` or `AugmentedTensor`. If this is
            `AugmentedTensor`, the `get_view` method will be call with default
            parameters.
        exclude : list
            List of ' `AugmentedTensor` label to exclude from the final view.
        size : int
            Size of each element of the grid

        Returns
        -------
        view: View
        """
        _views = [v for v in views if isinstance(v, View)]
        if len(_views) > 0:
            return View(Renderer.get_grid_view(_views, grid_size=None, cell_grid_size=size, add_title=add_title, **kwargs), title=title)

        # Include type
        include_type = [
            (type(l), ln, sn) for l, ln, sn in self._flatten_child([v for v in views if not isinstance(v, View)])
        ]
        # Exclude type
        exclude_type = [
            (type(l), ln, sn) for l, ln, sn in self._flatten_child([v for v in exclude if not isinstance(v, View)])
        ]

        def _is_include(label, set_name):
            if len(include_type) == 0:
                return True
            for t_label, t_label_name, t_set_name in include_type:
                if t_set_name == set_name and t_label == type(label):
                    return True
            return False

        def _is_exclude(label, set_name):
            if len(exclude_type) == 0:
                return False
            for t_label, t_label_name, t_set_name in exclude_type:
                if t_set_name == set_name and t_label == type(label):
                    return True
            return False

        def __get_view(sa_tensor: SpatialAugmentedTensor, info={}):
            n_views = []
            if sa_tensor.names[0] == "T" or sa_tensor.names[0] == "B":
                for t in range(0, sa_tensor.shape[0]):
                    n_info = {k: info[k] for k in info}
                    n_info.update({str(sa_tensor.names[0]): t})
                    n_views += __get_view(sa_tensor[t], info=n_info)
            else:
                if _is_include(self, None) and not _is_exclude(self, None):
                    __kwargs = inspect.getfullargspec(sa_tensor.__get_view__).args
                    __kwargs = {key: kwargs.pop(key) for key in __kwargs if key in kwargs}
                    view = sa_tensor.__get_view__(**__kwargs)
                    view.title = ",".join(f"{k}:{info[k]}" for k in info)
                    n_views.append(view)
                for label, label_name, set_name in sa_tensor._flatten_children():
                    # Do not render this label ?
                    if not _is_include(label, set_name) or _is_exclude(label, set_name):
                        continue
                    view_name = ",".join(f"{k}:{info[k]}" for k in info)
                    view_name += f", label_name={label_name}, set_name={set_name}"
                    if "frame" in inspect.getfullargspec(label.get_view).args:
                        view = label.get_view(frame=sa_tensor, **kwargs)
                    else:
                        view = label.get_view(**kwargs)
                    if view is not None:
                        view.title = view_name
                        n_views.append(view)
            return n_views

        n_views = __get_view(self)

        if self.names[0] == "T" or self.names[0] == "B":
            grid_size = max(1, len(n_views) // self.shape[0]) if grid_size is None else grid_size
        else:
            grid_size = None

        view = Renderer.get_grid_view(n_views, grid_size=grid_size, cell_grid_size=size, add_title=add_title, **kwargs)
        return View(view, title=title)

    def relative_to_absolute(self, x, dim, assert_integer=False, warn_non_integer=False):
        dim = dim.lower()
        assert dim in ["h", "w"], "dim should be 'h' or 'w'"
        ref = self.H if dim == "h" else self.W
        x *= ref

        if assert_integer:
            assert x.is_integer(), f"relative coordinates {x} have produced non-integer absolute coordinates"

        if warn_non_integer and not x.is_integer():
            warnings.warn(f"relative coordinates {x} have produced non-integer absolute coordinates")

        return round(x)

    def temporal(self, dim=None):
        """Add a temporal dimension on the tensor

        Parameters
        ----------
        dim : int or None
            The dim on which to add the temporal dimension. Can be 0 or 1 or None.
            None automatically determine position of T dim in respect of convention.

        Returns
        -------
        temporal_frame: aloscene.Frame
            A frame with a temporal dimension
        """
        if "T" in self.names:  # Already a temporal frame
            return self

        if dim is None:
            if self.names[0] == "B":
                dim = 1
            elif "B" in self.names:
                raise Exception(
                    "Cannot autodetermine temporal dimension position : tensor doesn't follow convention (B, T, ...) )"
                )
            else:
                dim = 0

        def set_n_names(names):
            pass

        tensor = self.rename(None)
        tensor = torch.unsqueeze(tensor, dim=dim)
        n_names = list(tensor._saved_names)
        n_names.insert(dim, "T")
        tensor.rename_(*tuple(n_names))

        def batch_label(tensor, label, name):
            if tensor._child_property[name]["mergeable"]:
                n_label_names = list(label._saved_names)
                n_label_names.insert(dim, "T")
                label.rename_(*tuple(n_label_names))
                for sub_name in label._children_list:
                    sub_label = getattr(label, sub_name)
                    if sub_label is not None:
                        self.apply_on_child(sub_label, lambda l: batch_label(label, l, sub_name), on_list=False)
            else:
                self.apply_on_child(label, lambda l: l.reset_names(), on_list=True)

        # Add a batch dimension on the
        for name in tensor._children_list:
            label = getattr(tensor, name)
            if label is not None:
                self.apply_on_child(label, lambda l: batch_label(tensor, l, name), on_list=False)

        return tensor

    def batch(self, dim=0):
        """Add a batch dimension on the tensor

        Parameters
        ----------
        dim : int
            The dim on which to add the batch dimension. Can be 0 or 1.

        Returns
        -------
        batch_frame: aloscene.Frame
            A frame with a batch dimension
        """
        if "B" in self.names:  # Already a temporal frame
            return self

        tensor = self.rename(None)
        tensor = torch.unsqueeze(tensor, dim=0)

        n_names = list(tensor._saved_names)
        n_names.insert(dim, "B")
        tensor.rename_(*tuple(n_names))

        def batch_label(tensor, label, name):
            """
            Recursively reset `label` names to take the new batch dimension into account:
            - If label is mergeable, "B" is added to its previous names
            - else, the previous name are restored.
            """
            if tensor._child_property[name]["mergeable"]:
                n_label_names = list(label._saved_names)
                n_label_names.insert(dim, "B")
                label.rename_(*tuple(n_label_names))
                for sub_name in label._children_list:
                    sub_label = getattr(label, sub_name)
                    if sub_label is not None:
                        self.apply_on_child(sub_label, lambda l: batch_label(label, l, sub_name), on_list=False)
            else:
                self.apply_on_child(label, lambda l: l.reset_names(), on_list=True)

        # Add a batch dimension on the label
        for name in tensor._children_list:
            label = getattr(tensor, name)
            if label is not None:
                self.apply_on_child(label, lambda l: batch_label(tensor, l, name), on_list=False)

        return tensor

    @staticmethod
    def batch_list(sa_tensors: list, pad_boxes: bool = False, pad_points2d: bool = False, intersection=False):
        """Given a list of Spatial Augmeted tensor of potentially different size (otherwise cat is enough) this method
        will create a new set of frame of same size (the max size across the frames)
        and add to each frame a mask with 1. on the padded area.

        Note that in order to work proprly, all attached label must implement the `pad` method.

        Parameters
        ----------
        sa_tensors: list or dict
            List of any aloscene.tensors.SpatialAugmentedTensor. If dict is given, this method will be applied on each
            list of spatial augmented tensors within the list
        pad_boxes: bool
            By default, do not rescale the boxes attached to the sptial augmented Tensor (see explanation in boxes2d.pad)
        pad_point2d: bool
            By default, False
        intersection: bool
            By default, an error is thrown if the tensors do not have the same children.
                Example1 : two frames from which only one has flow label.
                Example2 : two frames with different values for baseline property.
            If intersection is True, the batched tensor will have only children that exist in all original tensors.
            The properties with different values will be set to None.

        Returns
        -----------
        aloscene.tensors.SpatialAugmentedTensor
            A child of aloscene.tensors.SpatialAugmentedTensor (or dict of SpatialAugmentedTensor)
            with `mask` label to keep track of the padded areas.
        """
        assert len(sa_tensors) >= 1 and isinstance(sa_tensors, list)
        frame0 = sa_tensors[0]

        if isinstance(frame0, dict):
            DL = LDtoDL(sa_tensors)
            dict_of_sa = {key: SpatialAugmentedTensor.batch_list(val) for key, val in DL.items()}
            return dict_of_sa

        # Gather info on the given frames
        max_h, max_w = 0, 0
        dtype = sa_tensors[0].dtype
        device = sa_tensors[0].device

        # Retrieve the target size
        for i, frame in enumerate(sa_tensors):
            if frame is not None:
                max_h, max_w = max(frame.H, max_h), max(frame.W, max_w)

        n_sa_tensors = []
        for i, n_frame in enumerate(sa_tensors):
            if n_frame is None:
                continue
            # Add the batch dimension and drop the labels
            n_sa_tensors.append(n_frame.batch())
            frame = n_frame

        batch_size = len(n_sa_tensors)
        # Retrieve the new shapes and dim names
        n_tensor_shape, n_mask_shape = list(n_sa_tensors[0].shape), list(n_sa_tensors[0].shape)
        # New target frame size
        n_tensor_shape[0], n_mask_shape[0] = batch_size, batch_size
        n_tensor_shape[n_sa_tensors[0].names.index("H")] = max_h
        n_tensor_shape[n_sa_tensors[0].names.index("W")] = max_w
        # New target mask shape
        n_mask_shape[n_sa_tensors[0].names.index("H")] = max_h
        n_mask_shape[n_sa_tensors[0].names.index("W")] = max_w
        n_mask_shape[n_sa_tensors[0].names.index("C")] = 1
        n_names = n_sa_tensors[0].names

        n_padded_list = []
        for spatial_tensor in n_sa_tensors:
            h_pad = (0, max_h - spatial_tensor.H)
            w_pad = (0, max_w - spatial_tensor.W)
            padded_spatial_tensor = spatial_tensor.pad(h_pad, w_pad, pad_boxes=pad_boxes, pad_points2d=pad_points2d)
            n_padded_list.append(padded_spatial_tensor)

        # batch the tensors witch torch.cat ; set flag to specify desired behavior of torch.cat
        # necessary because torch.cat cannot accept kwargs (like intersection) that are not in the original signature
        intersect_old_value = AugmentedTensor.BATCH_LIST_INTERSECT
        AugmentedTensor.BATCH_LIST_INTERSECT = intersection
        n_augmented_tensors = torch.cat(n_padded_list, dim=0)
        AugmentedTensor.BATCH_LIST_INTERSECT = intersect_old_value

        # Set the new mask and tensor buffer filled up with zeros and ones
        # Also, for normalized frames, the zero value is actually different based on the mean/std
        n_tensor = torch.zeros(tuple(n_tensor_shape), dtype=dtype, device=device)

        # Create the batch list mask
        n_mask = torch.ones(tuple(n_mask_shape), dtype=torch.float, device=device)
        for b, frame in enumerate(n_sa_tensors):
            n_slice = frame.get_slices({"B": b, "H": slice(None, frame.H), "W": slice(None, frame.W)})
            n_tensor[n_slice].copy_(frame[0])
            n_mask[n_slice] = 0

        # n_frame = instance(n_tensor, names=n_names, normalization=normalization, device=device, mean_std=mean_std)
        n_augmented_tensors.append_mask(aloscene.Mask(n_mask, names=n_names))

        return n_augmented_tensors

    def _relative_to_absolute_hs_ws(self, hs=None, ws=None, assert_integer=True, warn_non_integer=False):
        """
        Parameters
        ----------
        hs : list or tuple of floats
            relative values for h dimension
        ww : list or tuple of floats
            relative values for w dimension
        """
        assert hs is not None or ws is not None, "ws and hs can't be both None"
        assert hs is None or isinstance(hs, (list, tuple)), "hs should be a list or a tuple of floats"
        assert ws is None or isinstance(ws, (list, tuple)), "ws should be a list or a tuple of floats"
        if hs is not None:
            hs = [self.relative_to_absolute(h, "H", assert_integer=assert_integer, warn_non_integer=warn_non_integer) for h in hs]
        if ws is not None:
            ws = [self.relative_to_absolute(w, "W", assert_integer=assert_integer, warn_non_integer=warn_non_integer) for w in ws]
        return hs, ws

    def _hflip_label(self, label, **kwargs):
        """
        Returns label horizontally flipped if possible, else unmodified label.
        """
        try:
            label_flipped = label._hflip(
                frame_size=self.HW, cam_intrinsic=self.cam_intrinsic, cam_extrinsic=self.cam_extrinsic, **kwargs
            )
        except AttributeError:
            return label
        else:
            return label_flipped

    def _vflip_label(self, label, **kwargs):
        """
        Returns label vertically flipped if possible, else unmodified label.
        """
        try:
            label_flipped = label._vflip(
                frame_size=self.HW, cam_intrinsic=self.cam_intrinsic, cam_extrinsic=self.cam_extrinsic, **kwargs
            )
        except AttributeError:
            return label
        else:
            return label_flipped

    def _crop_label(self, label, H_crop, W_crop, **kwargs):
        """
        Nothing to do.

        the ._crop method of SpatialAugmentedTensor crops the tensor
        by using the bracket notation tensor[hmin:hmax, wmin:wmax],
        which then triggers tensor._getitem_child,
        which calls label.crop on each labels.

        Therefore, if anything was done in _crop_label,
        it would be redundant with calling label.crop(...)
        """
        return label

    def _hflip(self, **kwargs):
        """Flip SpatialAugmentedTensor horizontally, but not its labels

        Parameters
        ----------


        Returns
        -------
        flipped_mask : aloscene.SpatialAugmentedTensor
            horizontally flipped SpatialAugmentedTensor
        """

        assert self.names[-2] == "H" and self.names[-1] == "W", f"expected format: […, H, W], got: {self.names}"
        return F.hflip(self.rename(None)).reset_names()

    def _vflip(self, **kwargs):
        """Flip SpatialAugmentedTensor vertically, but not its labels

        Parameters
        ----------


        Returns
        -------
        flipped_mask : aloscene.SpatialAugmentedTensor
            vertically flipped SpatialAugmentedTensor
        """

        assert self.names[-2] == "H" and self.names[-1] == "W", f"expected format: […, H, W], got: {self.names}"
        return F.vflip(self.rename(None)).reset_names()

    def _resize(self, size, interpolation=InterpolationMode.BILINEAR, **kwargs):
        """Resize SpatialAugmentedTensor, but not its labels

        Parameters
        ----------
        size : tuple of float
            target size (H, W) in relative coordinates between 0 and 1

        Returns
        -------
        resized : aloscene.SpatialAugmentedTensor
            resized tensor
        """

        assert self.names[-2] == "H" and self.names[-1] == "W", f"expected format: […, H, W], got: {self.names}"
        h = self.relative_to_absolute(size[0], "h", assert_integer=False)
        # assert_integer = False to avoid raising Assertion. Ex: when coordinates = 900.0000000000001
        w = self.relative_to_absolute(size[1], "w", assert_integer=False)
        # If a SpatialAgumentedTensor is empty, resize operation does not work. Use view instead.
        if ("N" in self.names and self.size("N") == 0) or ("C" in self.names and self.size("C") == 0):
            shapes = list(self.shape)[:-2] + [h, w]
            return self.rename(None).view(shapes).reset_names()
        return F.resize(self.rename(None), (h, w), interpolation=interpolation, antialias=True).reset_names()

    def _rotate(self, angle, center=None,**kwargs):
        """Rotate SpatialAugmentedTensor, but not its labels

        Parameters
        ----------
        angle : float
        center : list or tuple of coordinates in absolute format. Default is the center of the image


        Returns
        -------
        rotated : aloscene.SpatialAugmentedTensor
            rotated tensor
        """
        # If a SpatialAgumentedTensor is empty, rotate operation does not work. Use view instead.
        assert not (
            ("N" in self.names and self.size("N") == 0) or ("C" in self.names and self.size("C") == 0)
        ), "rotation is not possible on an empty tensor"
        return F.rotate(self.rename(None), angle,center=center).reset_names()

    def _crop(self, H_crop: tuple, W_crop: tuple, warn_non_integer=True, **kwargs):
        """Crop the SpatialAugmentedTensor

        Parameters
        ----------
        H_crop: tuple
            (start, end) between 0 and 1
        W_crop: tuple
            (start, end) between 0 and 1
        warn_non_integer: bool
            If True, warn if the crop is not integer

        Returns
        -------
        cropped sa_tensor: aloscene.SpatialAugmentedTensor
            cropped SpatialAugmentedTensor
        """

        H_crop, W_crop = self._relative_to_absolute_hs_ws(H_crop, W_crop, assert_integer=False, warn_non_integer=warn_non_integer)
        hmin, hmax = H_crop
        wmin, wmax = W_crop
        slices = self.get_slices({"H": slice(hmin, hmax), "W": slice(wmin, wmax)})
        return self[slices]

    def _pad_label(self, label, offset_y, offset_x, **kwargs):
        kwargs["frame_size"] = self.HW
        try:
            label_pad = label._pad(offset_y, offset_x, **kwargs)
            return label_pad
        except AttributeError:
            return label

    def _pad(self, offset_y: tuple, offset_x: tuple, **kwargs):
        """Pad the based on the given offset

        Parameters
        ----------
        offset_y: tuple
            (percentage top_offset, percentage bottom_offset) Percentage based on the previous size
        offset_x: tuple
            (percentage left_offset, percentage right_offset) Percentage based on the previous size

        Returns
        -------
            padded tensor
        """
        pad_top = int(round(offset_y[0] * self.H))
        pad_bottom = int(round(offset_y[1] * self.H))
        pad_left = int(round(offset_x[0] * self.W))
        pad_right = int(round(offset_x[1] * self.W))

        padding = [pad_left, pad_top, pad_right, pad_bottom]

        tensor_padded = F.pad(
            self.rename(None),
            padding,
            fill=kwargs.get("fill", 0),
            padding_mode=kwargs.get("padding_mode", "constant"),
        ).reset_names()

        return tensor_padded

    def _getitem_child(self, label, label_name, idx):
        """
        This method is used in AugmentedTensor.__getitem__
        The following must be specific to spatial labeled tensor only.
        """

        def _slice_list(label_list, curr_dim_idx, dim_idx, slicer):

            if isinstance(label_list, torch.Tensor):
                assert self._child_property[label_name]["mergeable"]
                n_slice = label_list.get_slices({label_list.names[dim_idx]: slicer}, label_list)
                return label_list[n_slice]
            if curr_dim_idx != dim_idx:
                n_label_list = []
                for l, label in enumerate(label_list):
                    n_label_list.append(_slice_list(label, curr_dim_idx + 1, dim_idx, slicer))
            else:
                return label_list[slicer]
            return n_label_list

        if isinstance(idx, tuple) or isinstance(idx, list):

            hw_crop = [None, None]

            dim_idx = 0
            label_dim_idx = 0

            for slicer_idx, slicer in enumerate(idx):

                if isinstance(slicer, type(Ellipsis)):
                    dim_idx += len(self.names) - len(idx[slicer_idx:]) + 1
                    label_dim_idx += len(self.names) - len(idx[slicer_idx:]) + 1
                elif isinstance(slicer, slice) and (slicer.start != None or slicer.stop != None):
                    if self.names[dim_idx] == "H":
                        hw_crop[0] = (slicer.start / self.H, slicer.stop / self.H)
                    elif self.names[dim_idx] == "W":
                        hw_crop[1] = (slicer.start / self.W, slicer.stop / self.W)
                    else:
                        allow_dims = self._child_property[label_name]["align_dim"]
                        if self.names[label_dim_idx] not in allow_dims:
                            raise Exception(
                                "Only a slice on the following none spatial dim is allow: {}. Trying to slice on {} for names {}".format(
                                    allow_dims, dim_idx, self.names
                                )
                            )
                        label = _slice_list(label, 0, label_dim_idx, slicer)
                    dim_idx += 1
                    label_dim_idx += 1
                elif isinstance(slicer, slice) and (slicer.start == None or slicer.stop == None):
                    dim_idx += 1
                    label_dim_idx += 1
                elif isinstance(slicer, int):
                    allow_dims = self._child_property[label_name]["align_dim"]
                    if self.names[dim_idx] not in allow_dims:
                        raise Exception(
                            "Only a slice on the following none spatial dim is allow: {}. Trying to slice on {} for names {}".format(
                                allow_dims, dim_idx, self.names
                            )
                        )
                    label = _slice_list(label, 0, label_dim_idx, slicer)
                    dim_idx += 1
                else:
                    raise Exception("Do not handle this slice")

            if hw_crop[0] is not None or hw_crop[1] is not None:
                label = self.apply_on_child(
                    label,
                    lambda l: l.crop(
                        hw_crop[0],
                        hw_crop[1],
                        frame_size=(self.H, self.W),
                        cam_intrinsic=self.cam_intrinsic,
                        cam_extrinsic=self.cam_extrinsic,
                    ),
                )

            return label
        else:
            return label[idx]
