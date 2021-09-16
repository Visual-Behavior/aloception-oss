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
    @staticmethod
    def __new__(
        cls,
        x,
        *args,
        cam_intrinsic: CameraIntrinsic = None,
        cam_extrinsic: CameraExtrinsic = None,
        mask=None,
        **kwargs,
    ):
        tensor = super().__new__(cls, x, *args, **kwargs)
        # Add camera parameters as labels
        tensor.add_label("cam_intrinsic", cam_intrinsic, align_dim=["B", "T"], mergeable=True)
        tensor.add_label("cam_extrinsic", cam_extrinsic, align_dim=["B", "T"], mergeable=True)
        tensor.add_label("mask", mask, align_dim=["B", "T"], mergeable=True)
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
        self._append_label("mask", mask, name)

    def append_cam_intrinsic(self, cam_intrinsic: CameraIntrinsic):
        self._append_label("cam_intrinsic", cam_intrinsic)

    def append_cam_extrinsic(self, cam_extrinsic: CameraExtrinsic):
        self._append_label("cam_extrinsic", cam_extrinsic)

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
            (type(l), ln, sn) for l, ln, sn in self._flatten_label([v for v in views if not isinstance(v, View)])
        ]
        # Exclude type
        exclude_type = [
            (type(l), ln, sn) for l, ln, sn in self._flatten_label([v for v in exclude if not isinstance(v, View)])
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
                for label, label_name, set_name in sa_tensor._flatten_labels():
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
            if tensor._label_property[name]["mergeable"]:
                label.rename_(*tuple(["T"] + list(label._saved_names)))
                for sub_name in label._labels_list:
                    sub_label = getattr(label, sub_name)
                    if sub_label is not None:
                        self.apply_on_label(sub_label, lambda l: batch_label(label, l, sub_name), on_list=False)
            else:
                self.apply_on_label(label, lambda l: l.reset_names(), on_list=True)

        # Add a batch dimension on the
        for name in tensor._labels_list:
            label = getattr(tensor, name)
            if label is not None:
                self.apply_on_label(label, lambda l: batch_label(tensor, l, name), on_list=False)

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
            if tensor._label_property[name]["mergeable"]:
                label.rename_(*tuple(["B"] + list(label._saved_names)))
                for sub_name in label._labels_list:
                    sub_label = getattr(label, sub_name)
                    if sub_label is not None:
                        self.apply_on_label(sub_label, lambda l: batch_label(label, l, sub_name), on_list=False)
            else:
                self.apply_on_label(label, lambda l: l.reset_names(), on_list=True)

        # Add a batch dimension on the label
        for name in tensor._labels_list:
            label = getattr(tensor, name)
            if label is not None:
                self.apply_on_label(label, lambda l: batch_label(tensor, l, name), on_list=False)

        return tensor

    @staticmethod
    def batch_list(sa_tensors: list, pad_boxes: bool = False):
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

        if (
            "N" in sa_tensors[0].names
            or "C" not in sa_tensors[0].names
            or "H" not in sa_tensors[0].names
            or "W" not in sa_tensors[0].names
        ):
            raise Exception(
                "{} (with names: {}) as it is, does not seem to be mergeable using batch_list.".format(
                    type(sa_tensors[0]), sa_tensors[0].names
                )
            )

        # Retrieve the target size
        for i, frame in enumerate(sa_tensors):
            if frame is not None:
                max_h, max_w = max(frame.H, max_h), max(frame.W, max_w)

        saved_frame_labels = {}
        n_sa_tensors = []
        for i, n_frame in enumerate(sa_tensors):
            if n_frame is None:
                continue
            # Add the batch dimension and drop the labels
            n_sa_tensors.append(n_frame.batch())
            frame = n_frame
            labels = n_sa_tensors[i].get_labels()

            # Merge labels on the first dim (TODO, move on an appropriate method)
            # The following can be merge into an other method in the augmented_tensor class that do roughly the same thing
            for label_name in labels:
                if labels[label_name] is None:
                    continue
                if label_name not in saved_frame_labels:
                    saved_frame_labels[label_name] = (
                        {key: [] for key in labels[label_name]} if isinstance(labels[label_name], dict) else []
                    )
                if isinstance(labels[label_name], dict):
                    for key in labels[label_name]:
                        saved_frame_labels[label_name] = frame._merge_label(
                            labels[label_name][key],
                            label_name,
                            key,
                            saved_frame_labels[label_name],
                            {"dim": 0},
                            check_dim=False,
                        )
                else:
                    saved_frame_labels = frame._merge_label(
                        labels[label_name], label_name, label_name, saved_frame_labels, {"dim": 0}, check_dim=False
                    )

        # Merge all aligned tensors on the batch dimension
        # Same thing that above, the following code should directlt use an other method
        # that roughly do the same thing
        # TODO: Test the following code since I don't have access ot mergeable label right now
        for label_name in saved_frame_labels:
            if not frame._label_property[label_name]["mergeable"]:
                continue
            if isinstance(saved_frame_labels[label_name], dict):
                for key in saved_frame_labels[label_name]:
                    saved_frame_labels[label_name][key] = torch.cat(saved_frame_labels[label_name][key], dim=0)
            else:
                saved_frame_labels[label_name] = torch.cat(saved_frame_labels[label_name], dim=0)

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
            padded_spatial_tensor = spatial_tensor.pad(h_pad, w_pad, pad_boxes=pad_boxes)
            n_padded_list.append(padded_spatial_tensor)

        n_augmented_tensors = torch.cat(n_padded_list, dim=0)

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
        which then triggers tensor._getitem_label,
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

    def _pad_label(self, label, offset_y, offset_x, **kwargs):
        kwargs["frame_size"] = self.HW
        try:
            label_pad = label._pad(offset_y, offset_x, **kwargs)
            return label_pad
        except AttributeError:
            return label

    def _pad(self, offset_y: tuple, offset_x: tuple, value=0, **kwargs):
        """Pad the based on the given offset

        Parameters
        ----------
        offset_y: tuple
            (percentage top_offset, percentage bottom_offset) Percentage based on the previous size
        offset_x: tuple
            (percentage left_offset, percentage right_offset) Percentage based on the previous size

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
        return tensor_padded

    def _getitem_label(self, label, label_name, idx):
        """
        This method is used in AugmentedTensor.__getitem__
        The following must be specific to spatial labeled tensor only.
        """

        def _slice_list(label_list, curr_dim_idx, dim_idx, slicer):

            if isinstance(label_list, torch.Tensor):
                assert self._label_property[label_name]["mergeable"]
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
                        allow_dims = self._label_property[label_name]["align_dim"]
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
                    allow_dims = self._label_property[label_name]["align_dim"]
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
                label = self.apply_on_label(
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
