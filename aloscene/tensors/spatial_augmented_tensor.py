import matplotlib.pyplot as plt
import math
import torchvision.transforms.functional as F
import torch

from aloscene.renderer import View, Renderer
from .augmented_tensor import AugmentedTensor
import inspect
import aloscene
from aloscene.camera_calib import CameraExtrinsic, CameraIntrinsic
from aloscene.utils.data_utils import LDtoDL


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
    """

    @staticmethod
    def __new__(
        cls,
        x,
        *args,
        cam_intrinsic: CameraIntrinsic = None,
        cam_extrinsic: CameraExtrinsic = None,
        # For stereo setups
        camera_side: str = None,
        baseline: float = None,
        mask=None,
        **kwargs,
    ):
        tensor = super().__new__(cls, x, *args, **kwargs)

        tensor.add_child("mask", mask, align_dim=["B", "T"], mergeable=True)
        tensor.add_property("baseline", baseline)
        tensor.add_property("camera_side", camera_side)

        # Intrisic and extrinsic parameters are cloned by default, to prevent having multiple reference
        # of the same intrisic/extrinsic across nodes.
        cam_intrinsic = cam_intrinsic.clone() if cam_intrinsic is not None else cam_intrinsic
        cam_extrinsic = cam_extrinsic.clone() if cam_extrinsic is not None else cam_extrinsic
        tensor.add_child("cam_intrinsic", cam_intrinsic, align_dim=["B", "T"], mergeable=True)
        tensor.add_child("cam_extrinsic", cam_extrinsic, align_dim=["B", "T"], mergeable=True)

        return tensor

    def __init__(self, x, *args, **kwargs):
        super().__init__(x)

    @property
    def HW(self):
        return (self.H, self.W)

    @property
    def W(self):
        return self.shape[self.names.index("W")]

    @property
    def H(self):
        return self.shape[self.names.index("H")]

    def append_mask(self, mask, name: str = None):
        """Attach a mask to the frame.

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

    def get_view(self, views: list = [], exclude=[], size=None, grid_size=None, **kwargs):
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
            return View(Renderer.get_grid_view(_views, grid_size=None, cell_grid_size=size, **kwargs))

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

        view = Renderer.get_grid_view(n_views, grid_size=grid_size, cell_grid_size=size, **kwargs)
        return View(view)

    def relative_to_absolute(self, x, dim, assert_integer=False):
        dim = dim.lower()
        assert dim in ["h", "w"], "dim should be 'h' or 'w'"
        ref = self.H if dim == "h" else self.W
        x *= ref

        if assert_integer:
            assert x.is_integer(), f"relative coordinates {x} have produced non-integer absolute coordinates"
        return round(x)

    def temporal(self):
        """Add a temporal dimension on the tensor

        Returns
        -------
        temporal_frame: aloscene.Frame
            A frame with a temporal dimension
        """
        if "T" in self.names:  # Already a temporal frame
            return self

        tensor = self.rename(None)
        tensor = torch.unsqueeze(tensor, dim=0)
        tensor.rename_(*tuple(["T"] + list(tensor._saved_names)))

        def batch_label(tensor, label, name):
            if tensor._child_property[name]["mergeable"]:
                label.rename_(*tuple(["T"] + list(label._saved_names)))
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

    def batch(self):
        """Add a batch dimension on the tensor

        Returns
        -------
        batch_frame: aloscene.Frame
            A frame with a batch dimension
        """
        if "B" in self.names:  # Already a temporal frame
            return self

        tensor = self.rename(None)
        tensor = torch.unsqueeze(tensor, dim=0)
        tensor.rename_(*tuple(["B"] + list(tensor._saved_names)))

        def batch_label(tensor, label, name):
            """
            Recursively reset `label` names to take the new batch dimension into account:
            - If label is mergeable, "B" is added to its previous names
            - else, the previous name are restored.
            """
            if tensor._child_property[name]["mergeable"]:
                label.rename_(*tuple(["B"] + list(label._saved_names)))
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
    def batch_list(sa_tensors: list, pad_boxes: bool = False, pad_points2d: bool = False):
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
            padded_spatial_tensor = spatial_tensor.pad(
                h_pad, w_pad, pad_boxes=pad_boxes, pad_points2d=pad_points2d, padding_mask=True
            )
            n_padded_list.append(padded_spatial_tensor)

        n_augmented_tensors = torch.cat(n_padded_list, dim=0)

        return n_augmented_tensors

    def _relative_to_absolute_hs_ws(self, hs=None, ws=None, assert_integer=True):
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
            hs = [self.relative_to_absolute(h, "H", assert_integer) for h in hs]
        if ws is not None:
            ws = [self.relative_to_absolute(w, "W", assert_integer) for w in ws]
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

    def _resize(self, size, **kwargs):
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
        return F.resize(self.rename(None), (h, w)).reset_names()

    def _crop(self, H_crop: tuple, W_crop: tuple, **kwargs):
        """Crop the SpatialAugmentedTensor

        Parameters
        ----------
        H_crop: tuple
            (start, end) between 0 and 1
        W_crop: tuple
            (start, end) between 0 and 1

        Returns
        -------
        cropped sa_tensor: aloscene.SpatialAugmentedTensor
            cropped SpatialAugmentedTensor
        """
        H_crop, W_crop = self._relative_to_absolute_hs_ws(H_crop, W_crop, assert_integer=False)
        hmin, hmax = H_crop
        wmin, wmax = W_crop
        slices = self.get_slices({"H": slice(hmin, hmax), "W": slice(wmin, wmax)})
        return self[slices]


    def _get_paded_mask(self, offset_y: tuple, offset_x: tuple):
        """ Generate a mask to keep track of the padded part of the Spatial Tensor
        """

        # Set the new height and weight of the frame
        pad_top, pad_bottom, pad_left, pad_right = (
            round(offset_y[0] * self.H),
            round(offset_y[1] * self.H),
            round(offset_x[0] * self.W),
            round(offset_x[1] * self.W),
        )
        n_H = self.H + pad_top + pad_bottom
        n_W = self.W + pad_left + pad_right

        # Define the new shape of the padded frame
        n_mask_shape = list(self.shape)
        n_mask_shape[self.names.index("C")] = 1
        n_mask_shape[self.names.index("H")] = n_H
        n_mask_shape[self.names.index("W")] = n_W

        if self.mask is None:
            mask = torch.zeros(*tuple(n_mask_shape), dtype=self.dtype, device=self.device) + 1
            n_slice = self.get_slices({"H": slice(pad_top, n_H - pad_bottom), "W": slice(pad_left, n_W - pad_right)})
            mask[n_slice] = 0
        else:
            n_slice = self.get_slices({"H": slice(pad_top, n_H - pad_bottom), "W": slice(pad_left, n_W - pad_right)})
            mask = torch.zeros(*tuple(n_mask_shape), dtype=self.dtype, device=self.device) + 1
            mask[n_slice] = self.mask.as_tensor()


        mask = aloscene.Mask(mask, names=self.names)
        return mask

    def _pad_label(self, label, offset_y, offset_x, exception=None, **kwargs):
        # One particular label will not be pad (the mask label. This is a special case handle differently)
        if exception is not None and exception is label:
            return label
        kwargs["frame_size"] = self.HW
        try:
            label_pad = label._pad(offset_y, offset_x, **kwargs)
            return label_pad
        except AttributeError:
            return label

    def pad(self, offset_y: tuple, offset_x: tuple, **kwargs):
        """
        Pad AugmentedTensor, and its labels recursively

        Parameters
        ----------
        offset_y: tuple of float or tuple of int
            (percentage top_offset, percentage bottom_offset) Percentage based on the previous size If tuple of int
            the absolute value will be converted to float (percentahe) before to be applied.
        offset_x: tuple of float or tuple of int
            (percentage left_offset, percentage right_offset) Percentage based on the previous size. If tuple of int
            the absolute value will be converted to float (percentage) before to be applied.

        Returns
        -------
        croped : aloscene AugmentedTensor
            croped tensor
        """
        if isinstance(offset_y[0], int) and isinstance(offset_y[1], int):
            offset_y = (offset_y[0] / self.H, offset_y[1] / self.H)
        if isinstance(offset_x[0], int) and isinstance(offset_x[1], int):
            offset_x = (offset_x[0] / self.W, offset_x[1] / self.W)

        padded = self._pad(offset_y, offset_x, **kwargs)
        padded.recursive_apply_on_children_(lambda label: self._pad_label(
            label, offset_y, offset_x, exception=padded.mask, **kwargs)
        )
        return padded

    def _pad(self, offset_y: tuple, offset_x: tuple, value=0, padding_mask=False, **kwargs):
        """Pad the based on the given offset

        Parameters
        ----------
        offset_y: tuple
            (percentage top_offset, percentage bottom_offset) Percentage based on the previous size
        offset_x: tuple
            (percentage left_offset, percentage right_offset) Percentage based on the previous size
        padding_mask: bool
            If the padding_mask is True a mask will be stored on the frame to show the padded area.


        Returns
        -------
        padded
        """
        pad_top = int(round(offset_y[0] * self.H))
        pad_bottom = int(round(offset_y[1] * self.H))
        pad_left = int(round(offset_x[0] * self.W))
        pad_right = int(round(offset_x[1] * self.W))

        padding = [pad_left, pad_top, pad_right, pad_bottom]
        tensor_padded = F.pad(self.rename(None), padding, padding_mode="constant", fill=value).reset_names()

        if padding_mask and not isinstance(self, aloscene.Mask):
            mask = self._get_paded_mask(offset_y, offset_x)
            tensor_padded.mask = mask

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
