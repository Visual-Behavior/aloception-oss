import torch
import inspect
import numpy as np
from typing import *


class AugmentedTensor(torch.Tensor):
    """Tensor with attached labels"""

    # Common dim names that must be aligned to handle label on a Tensor.
    COMMON_DIM_NAMES = ["B", "T"]

    @staticmethod
    def __new__(cls, x, names=None, device=None, *args, **kwargs):
        # TODO The following is not optigal yet
        # I do it to be able to create directly an Aumented Tensor to the GPU
        # But the following Workaround first create the tensor on the CPU, then
        # move the tensor to the target device
        if isinstance(x, torch.Tensor) and (device != None or x.device != "cpu"):
            device = x.device if device is None else device

        if isinstance(x, torch.Tensor):
            tensor = super().__new__(cls, torch.ones(x.shape), *args, **kwargs)
        else:
            tensor = super().__new__(cls, x, *args, **kwargs)

        tensor._children_list = []
        tensor._child_property = {}
        tensor._property_list = []
        tensor.add_property("_saved_names", None)
        tensor.add_property("_auto_restore_names", False)

        if names is not None:
            tensor.rename_(*names)
        else:
            t = type(tensor)
            raise Exception(f"AugmentedTensor ({t}) method must be create with `names` dim.")

        if device is not None:
            tensor = tensor.to(device)

        if isinstance(x, torch.Tensor):
            tensor = tensor * x

        return tensor

    def __init__(self, x, **kwargs):
        super().__init__()

    def drop_children(self):
        """Remove all children from this augmented tensor and return
        the removed children.
        """
        labels = {}
        for name in self._children_list:
            labels[name] = {"value": getattr(self, name), "property": self._child_property[name]}
            setattr(self, name, None)
        return labels

    def get_children(self):
        """Get all children attached from this augmented tensor"""
        labels = {}
        for name in self._children_list:
            # Use apply to return the same label but with a new structure
            # so that if the returned structure is changed, this will not impact the current one
            labels[name] = {
                "value": self.apply_on_child(getattr(self, name), lambda l: l),
                "property": self._child_property[name],
            }
        return labels

    def set_children(self, labels):
        """Set children on this augmented tensor"""
        for name in labels:
            if name not in self._children_list:
                self.add_child(name, labels[name]["value"], **labels[name]["property"])
            else:
                setattr(self, name, labels[name]["value"])
                self._child_property[name] = labels[name]["property"]
        return labels

    @staticmethod
    def apply_on_child(label, func, on_list=True):
        """Apply a function on a child"""

        def _apply_on_child(label, func, on_list):
            result = None
            if isinstance(label, dict):
                result = {}
                for key in label:
                    result[key] = _apply_on_child(label[key], func, on_list=on_list)
            elif isinstance(label, list) and on_list:
                result = []
                for el in label:
                    result.append(_apply_on_child(el, func, on_list=on_list))
            elif isinstance(label, list) and not on_list:
                result = func(label)
            else:
                result = func(label)
            return result

        # deal with missing labels
        _func = lambda l: None if l is None else func(l)
        return _apply_on_child(label, _func, on_list)

    def _flatten_child(self, label, label_name=None, set_name=None):
        flatten_list = []
        if isinstance(label, dict):
            for key in label:
                flatten_list += self._flatten_child(label[key], label_name=label_name, set_name=key)
        elif isinstance(label, list):
            for el in label:
                flatten_list += self._flatten_child(el, label_name=label_name, set_name=set_name)
        elif label is not None:
            flatten_list += [(label, label_name, set_name)]
        return flatten_list

    def _flatten_children(self):
        """This method go through all label of the augmented tensor
        and return a flatten list of the labels.

        TODO: ? This method can be extend if needed to return the flatten version
        of only the desired label.

        Returns
        -------
        labels: list
            List of `Augmented Tensor`
        """
        labels = []
        for name in self._children_list:
            label = getattr(self, name)
            if label is not None:
                labels += self._flatten_child(label, label_name=name)
        return labels

    def add_property(self, var_name, variable):
        if var_name not in self._property_list:
            self._property_list.append(var_name)
        setattr(self, var_name, variable)

    def _check_child_name_alignment(self, variable):
        """Check label name _check_child_name_alignment"""

        def __check_child_name_alignment(var):
            if not any(v != None for v in self.names):
                return True
            elif not any(v in self.COMMON_DIM_NAMES for v in var.names):
                return True
            for dim_id, dim_name in enumerate(var.names):
                if (
                    dim_id < len(self.names)
                    and dim_name in self.COMMON_DIM_NAMES
                    and self.names[dim_id] == dim_name
                ):
                    return True
            raise Exception(
                f"Impossible to align label name dim ({var.names}) with the tensor name dim ({self.names})."
            )
            return False

        if variable is not None:
            self.apply_on_child(variable, __check_child_name_alignment)

        return True

    def add_child(self, child_name, child, align_dim=["B", "T"], mergeable=True, **kwargs):
        """Add/attached an augmented tensor on this augmented tensor.

        Parameters
        ----------
        child_name : str
            Name of the child to accahed
        augmented: aloscene.AugmentedTensor
            instance of Augmented tensor
        align_dim: list
            List that enumerate the dimensions the child is allign with. Align on "B" & "T" means that that
            parent child and the child child can be freely manipulate on theses dimensions without corrupting that
            data structure.
        mergeable: bool
            True by default. In some cases, if one want to merge parents, an appropriate stategy for the chils
            child must be choosen. If all child are guarentee to be the same size, theses children are set to be
            mergeable and wll be merged together within a new dimension.
            However, for some children, it might not be true. For instance, one frame can have a variable
            number of boxes ber frame. Therefore, merging frame with boxes children could not be possible. If this is the
            case, mergeable=False must be set. The children will then be merged within a list (instead of being merged
            as a new tensor.)
        """
        if child is not None:
            self._check_child_name_alignment(child)

        if child_name not in self._children_list:
            self._children_list.append(child_name)
            kwargs["align_dim"] = align_dim
            kwargs["mergeable"] = mergeable
            self._child_property[child_name] = kwargs

        setattr(self, child_name, child)

    def _append_child(self, child_name: str, child, set_name: str = None):
        """
        Attach a new value for a given child name.

        Parameters
        ----------
        child_name: str
            name of the child
        child:
            value given to the child
        set_name : str
            A `set_name` can be set for the value of this child.
            If None, the child value will be attached to the child name without a set (if possible).
        """
        assert isinstance(child_name, str)
        assert isinstance(set_name, str) or set_name == None
        label = getattr(self, child_name)
        class_name = type(self).__name__
        if label is not None and not isinstance(label, dict):
            raise Exception(
                f"This instance of {class_name} already has an unnamed label of type {child_name}."
                " Drop the unnamed label and add it back with name."
            )
        elif isinstance(label, dict) and set_name is None:
            raise Exception(
                "This instance of {class_name} already has named labels of type {child_name} attached to it."
                "New labels of the same time can only be added with a name."
            )

        if label is None and set_name is not None:
            label = {}
            setattr(self, child_name, label)

        if set_name is None:
            setattr(self, child_name, child)
        else:
            label[set_name] = child

    def _getitem_child(self, label, label_name, idx):
        """
        This method is used in AugmentedTensor.__getitem__
        The following must be specific to spatial labeled tensor only.
        """

        def _slice_list(label_list, curr_dim_idx, dim_idx, slicer):
            assert isinstance(label_list, list), f"label_list is not list:{label_list}"
            if curr_dim_idx != dim_idx:
                n_label_list = []
                for l, label in enumerate(label_list):
                    n_label_list.append(_slice_list(label, curr_dim_idx + 1, dim_idx, slicer))
            else:
                return label_list[slicer]
            return n_label_list

        if isinstance(idx, tuple) or isinstance(idx, list):
            label_dim_idx = 0
            for slicer_idx, slicer in enumerate(idx):

                if isinstance(slicer, type(Ellipsis)):
                    label_dim_idx += len(self.names) - len(idx[slicer_idx:]) + 1

                elif isinstance(slicer, slice) and (slicer.start != None or slicer.stop != None):
                    allow_dims = self._child_property[label_name]["align_dim"]
                    if self.names[label_dim_idx] not in allow_dims:
                        raise Exception(
                            "Only a slice on the following none spatial dim is allow: {}. Trying to slice on {} for names {}".format(
                                allow_dims, label_dim_idx, self.names
                            )
                        )
                    label = _slice_list(label, 0, label_dim_idx, slicer)
                    label_dim_idx += 1

                elif isinstance(slicer, slice) and (slicer.start == None or slicer.stop == None):
                    label_dim_idx += 1

                elif isinstance(slicer, int):
                    allow_dims = self._child_property[label_name]["align_dim"]
                    if self.names[label_dim_idx] not in allow_dims:
                        raise Exception(
                            "Only a slice on the following none spatial dim is allow: {}. Trying to slice on {} for names {}".format(
                                allow_dims, label_dim_idx, self.names
                            )
                        )
                    label = _slice_list(label, 0, label_dim_idx, slicer)

                else:
                    raise Exception("Do not handle this slice")
            return label
        else:
            if isinstance(idx, torch.Tensor) and hasattr(label, "reset_names"):
                return label.rename_(None)[idx]  # .reset_names()
            else:
                return label[idx]

    def __getitem__(self, idx):

        name_to_n_label = {}
        for name in self._children_list:
            label = getattr(self, name)
            if label is not None:
                name_to_n_label[name] = self.apply_on_child(
                    label, lambda l: self._getitem_child(l, name, idx), on_list=False
                )

        if isinstance(idx, torch.Tensor):
            if not idx.dtype == torch.bool:
                if torch.equal(idx ** 3, idx):
                    raise IndexError(f"Unvalid mask. Expected mask elements to be in [0, 1, True, False]")
            tensor = self * idx
        else:
            tensor = torch.Tensor.__getitem__(self, idx)

        for name in name_to_n_label:
            tensor.__setattr__(name, name_to_n_label[name])

        return tensor

    def __setattr__(self, key, value):
        # if hasattr(self, "_children_list") and  key in self._children_list and check:
        #    self._check_child_name_alignment(value)
        super().__setattr__(key, value)
        pass

    def clone(self, *args, **kwargs):
        n_frame = super().clone(*args, **kwargs)
        n_frame._property_list = self._property_list
        n_frame._children_list = self._children_list
        n_frame._child_property = self._child_property

        for name in self._property_list:
            setattr(n_frame, name, getattr(self, name))
        for name in self._children_list:
            setattr(n_frame, name, self.apply_on_child(getattr(self, name), lambda l: l.clone()))

        return n_frame

    def to(self, *args, **kwargs):
        """ """

        n_frame = super().to(*args, **kwargs)
        n_frame._property_list = self._property_list
        n_frame._children_list = self._children_list
        n_frame._child_property = self._child_property

        for name in self._property_list:
            setattr(n_frame, name, getattr(self, name))
        for name in self._children_list:

            device = None
            if len(args) >= 1:
                device = args[0]
            elif "device" in kwargs:
                device = kwargs["device"]

            # Keep the label on the same devie
            if isinstance(device, torch.device):
                label = getattr(self, name)
                if label is not None:
                    n_label = self.apply_on_child(label, lambda l: l.to(*args, **kwargs))
                    setattr(n_frame, name, n_label)
            else:
                label = getattr(self, name)
                setattr(n_frame, name, label)

        return n_frame

    def cpu(self, *args, **kwargs):
        """Send the current augmentend tensor on the cpu with all its labels
        recursively.
        """
        # Send the frame on the cpu and set back the property
        n_frame = super().cpu(*args, **kwargs)
        n_frame._property_list = self._property_list
        n_frame._children_list = self._children_list
        n_frame._child_property = self._child_property
        for name in self._property_list:
            setattr(n_frame, name, getattr(self, name))
        # Set back the labels
        for name in self._children_list:
            label = getattr(self, name)
            if label is not None:
                setattr(n_frame, name, self.apply_on_child(label, lambda l: l.cpu(*args, **kwargs)))
        return n_frame

    def cuda(self, *args, **kwargs):
        """Send the current augmentend tensor on cuda with all its labels
        recursively.
        """
        # Send the frame on cuda and set back the property
        n_frame = super().cuda(*args, **kwargs)
        n_frame._property_list = self._property_list
        n_frame._children_list = self._children_list
        n_frame._child_property = self._child_property
        for name in self._property_list:
            setattr(n_frame, name, getattr(self, name))
        # Set back the labels
        for name in self._children_list:
            label = getattr(self, name)
            if label is not None:
                setattr(
                    n_frame, name, self.apply_on_child(label, lambda l: l.cuda(*args, **kwargs))
                )
        return n_frame

    def _merge_child(self, label, label_name, key, dict_merge, kwargs, check_dim=True):

        target_dim = kwargs["dim"]
        if target_dim > 0 and self._child_property[label_name]["mergeable"]:
            # If the labels are mergable, we don't want to fill up the structure on the target_dim
            # if target_dim > 0. We just want to merge on 0 and then call the torch.cat method to
            # merge everything on the real target dimension.
            target_dim = 0

        if (
            check_dim
            and self.names[target_dim] not in self._child_property[label_name]["align_dim"]
        ):
            raise Exception(
                "Can only merge labeled tensor on the following dimension '{}'. \
                \nDrop the labels before to apply such operations or convert your labeled tensor to tensor first.".format(
                    self._child_property[label_name]["align_dim"]
                )
            )

        # Create the list structure to merge the label on the appropriate dimension
        def _create_dict_merge_structure(sub_label, dim, target_dim):
            n_dm = []
            if dim == target_dim:
                return n_dm
            else:
                for s in range(len(sub_label)):
                    n_dm.append([])
                    n_dm[-1] = _create_dict_merge_structure(sub_label[s], dim + 1, target_dim)
            return n_dm

        if key not in dict_merge or (target_dim > 0 and len(dict_merge[key]) == 0):
            dict_merge[key] = _create_dict_merge_structure(label, 0, target_dim)

        # Fill up the structure on the appropriate dimension
        def _fillup_dict(dm, sub_label, dim, target_dim):
            if dim == target_dim:
                # Append both list together
                if isinstance(sub_label, list):
                    dm += sub_label
                else:
                    dm.append(sub_label)
            else:
                for s in range(len(sub_label)):
                    _fillup_dict(dm[s], sub_label[s], dim + 1, target_dim)

        _fillup_dict(dict_merge[key], label, 0, target_dim)

        return dict_merge

    def _merge_tensor(self, n_tensor, tensor_list, func, types, args=(), kwargs=None):
        """Merge tensors together and their associated labels"""
        labels_dict2list = {}
        # Setup the new structure before to merge
        prop_name_to_value = {}

        for tensor in tensor_list:

            for prop in tensor._property_list:
                if prop in prop_name_to_value:
                    assert prop_name_to_value[prop] == getattr(
                        tensor, prop
                    ), f"Trying to merge augmented tensor with different property: {prop}, {prop_name_to_value[prop]}, {getattr(tensor, prop)}"
                prop_name_to_value[prop] = getattr(tensor, prop)

            for label_name in tensor._children_list:
                label_value = getattr(tensor, label_name)
                if label_value is not None and isinstance(label_value, dict):
                    labels_dict2list[label_name] = {}
                elif label_value is not None:
                    labels_dict2list[label_name] = []

        for tensor in tensor_list:
            if isinstance(tensor, type(self)):
                for label_name in tensor._children_list:
                    label_value = getattr(tensor, label_name)
                    if label_name not in labels_dict2list:
                        continue
                    if isinstance(label_value, dict):
                        for key in label_value:
                            labels_dict2list[label_name] = self._merge_child(
                                label_value[key],
                                label_name,
                                key,
                                labels_dict2list[label_name],
                                kwargs,
                            )
                    elif label_value is None and isinstance(labels_dict2list[label_name], dict):
                        for key in labels_dict2list[label_name]:
                            labels_dict2list[label_name] = self._merge_child(
                                label_value, label_name, key, labels_dict2list[label_name], kwargs
                            )
                    else:
                        self._merge_child(
                            label_value, label_name, label_name, labels_dict2list, kwargs
                        )
            else:
                raise Exception("Can't merge none AugmentedTensor with AugmentedTensor")

        # Merge all labels together
        for label_name in labels_dict2list:
            if self._child_property[label_name]["mergeable"]:
                if isinstance(labels_dict2list[label_name], dict):
                    for key in labels_dict2list[label_name]:
                        args = list(args)
                        args[0] = labels_dict2list[label_name][key]
                        labels_dict2list[label_name][key] = func(*tuple(args), **kwargs)
                else:
                    args = list(args)
                    args[0] = labels_dict2list[label_name]
                    labels_dict2list[label_name] = func(*tuple(args), **kwargs)
                setattr(n_tensor, label_name, labels_dict2list[label_name])
            else:
                setattr(n_tensor, label_name, labels_dict2list[label_name])

    def _squeeze_unsqueeze_dim(self, tensor, func, types, squeeze, args=(), kwargs=None):
        """This callback is called when the labeled_tensor
        is call.
        """
        dim = kwargs["dim"] if "dim" in kwargs else 0

        if dim != 0 and dim !=1:
            raise Exception(
                f"Impossible to expand the labeld tensor on the given dim: {dim}. Export your labeled tensor into tensor before to do it."
            )
        # elif self._saved_names[dim] == "B" and not squeeze:
        #    raise Exception("Impossible to expand the labeled tensor beyond the batch dimension fow now. Export your labeled tensor into tensor before to do it.")

        def _handle_expand_on_label(label, name):
            if not squeeze:
                if self._child_property[name]["mergeable"]:
                    return label[None]
                else:
                    return [label]
            else:
                return label[0]

        for name in self._children_list:
            label = getattr(tensor, name)
            if label is not None:
                results = self.apply_on_child(
                    label, lambda l: _handle_expand_on_label(l, name), on_list=False
                )
                setattr(tensor, name, results)

    def __iter__(self):
        for t in range(len(self)):
            yield self[t]

    def __torch_function__(self, func, types, args=(), kwargs=None):
        def _merging_frame(args):
            if len(args) >= 1 and isinstance(args[0], list):
                for el in args[0]:
                    if isinstance(el, type(self)):
                        return True
                return False
            return False

        if kwargs is None:
            kwargs = {}

        if func.__name__ == "__reduce_ex__":
            self.rename_(None, auto_restore_names=True)
            tensor = super().__torch_function__(func, types, args, kwargs)
        else:
            tensor = super().__torch_function__(func, types, args, kwargs)

        if isinstance(tensor, type(self)):

            tensor._property_list = self._property_list
            tensor._children_list = self._children_list
            tensor._child_property = self._child_property
            for name in self._property_list:
                setattr(tensor, name, getattr(self, name))

            # Some method is apply to merge some frame
            # Try to merge the frames components as well and
            # TODO: check that all frame have the same parameters.
            if _merging_frame(args):
                self._merge_tensor(tensor, args[0], func, types, args=args, kwargs=kwargs)

            for name in self._children_list:
                if not hasattr(
                    tensor, name
                ):  # Set what is not already set (some could have been set by the merge function above)
                    setattr(tensor, name, getattr(self, name))

            # The torch method called expand the shape of the tensor.
            # Check how to extand the label based on this operation (if possible)
            if len(self.shape) < len(tensor.shape):
                self._squeeze_unsqueeze_dim(tensor, func, types, False, args, kwargs)
            elif len(self.shape) > len(tensor.shape):
                self._squeeze_unsqueeze_dim(tensor, func, types, True, args, kwargs)

            return tensor
        else:
            return tensor

    def reset_names(self):
        def _reset_names(l):
            if l is None:
                return None
            elif not hasattr(l, "reset_names"):
                return l
            else:
                return l.reset_names()

        for name in self._children_list:
            label = getattr(self, name)
            if label is not None:
                self.apply_on_child(label, _reset_names)

        if self._saved_names is not None and any(v is not None for v in self._saved_names):
            self_ref_tensor = self.rename_(*self._saved_names)
            self_ref_tensor._saved_names = None
            return self_ref_tensor
        else:
            return self

    def rename_(self, *args, auto_restore_names=False, **kwargs):
        """Rename the dimensions of your augmented Tensor.

        Parameters
        ----------
        auto_restore_names: bool
            Somehow hacky but this is maybe the only way to pass Augmented
            tensor through the datapipeline when doing multi-gpu training. Looking
            forward for the next pytorch release.
            Therefore, when passing your Augmented Tensor through a Datapipeline (the output of __getitem__)
            you must rename your augmented tensor using `augmented_tensor.rename(auto_restore=True)`.

            Note: This this not needed for the datasets that inhert from alodataset.BaseDataset
        """

        def _rename(v):
            if v is None:
                return v
            if "auto_restore_names" in inspect.getfullargspec(v.rename_).kwonlyargs:
                return v.rename_(None, auto_restore_names=auto_restore_names)
            else:
                return v.rename_(None)

        if args[0] is None:
            for name in self._children_list:
                label = getattr(self, name)
                if label is not None:
                    self.apply_on_child(label, _rename)
        self._saved_names = self.names
        super().rename_(*args, **kwargs)
        self._auto_restore_names = auto_restore_names
        return self

    def rename(self, *args, auto_restore_names=False, **kwargs):
        """Rename the dimensions of your augmented Tensor.

        Parameters
        ----------
        auto_restore_names: bool
            Somehow hacky but this is maybe the only way to pass Augmented
            tensor through the datapipeline when doing multi-gpu training. Looking
            forward for the next pytorch release.
            Therefore, when passing your Augmented Tensor through a Datapipeline (the output of __getitem__)
            you must rename your augmented tensor using `augmented_tensor.rename(auto_restore=True)`.

            Note: This this not needed for the datasets that inhert from alodataset.BaseDataset
        """

        def _rename(v):
            if v is None:
                return v
            if "auto_restore_names" in inspect.getfullargspec(v.rename).kwonlyargs:
                return v.rename(None, auto_restore_names=auto_restore_names)
            else:
                return v.rename(None)

        saved_names = self.names
        tensor = super().rename(*args, **kwargs)

        if args[0] is None:
            for name in self._children_list:
                label = getattr(self, name)
                if label is not None:
                    setattr(tensor, name, self.apply_on_child(label, _rename))

        tensor._saved_names = saved_names
        tensor._auto_restore_names = auto_restore_names
        return tensor

    def as_tensor(self):
        """Returns a new tensor based on the same memory location than the current
        Augmted Tensor

        Returns
        -------
        n_tensor: torch.Tensor
            A new tensor based on the same content, same dtype, same device
        """
        n_tensor = self.clone()
        n_tensor.__class__ = torch.Tensor
        n_tensor.rename_(None)
        return n_tensor

    def as_numpy(self, dtype=np.float16):
        """Returns numpy array on the cpu

        Parameters
        ----------
            dtype: (: np.dtype)
                The output dtype. Default np.float16.
        """
        tensor = self
        return tensor.detach().cpu().contiguous().numpy().astype(dtype)

    def __repr__(self):
        _str = self.as_tensor().__repr__()
        n_str = f"tensor(\n\t"

        for name in self._property_list:
            if name.startswith("_") or getattr(self, name) is None:
                continue
            v = getattr(self, name)
            n_str += f"{name}={v}, "
        n_str += "\n\t"

        for name in self._children_list:
            values = getattr(self, name)
            if values is None:
                continue
            if isinstance(values, dict):
                content_value = ""
                for key in values:
                    if isinstance(values[key], list):
                        cvalue = (
                            f"{key}:["
                            + ", ".join(
                                [
                                    "{}".format(len(k) if k is not None else None)
                                    for k in values[key]
                                ]
                            )
                            + "]"
                        )
                        content_value += f"{cvalue}, "
                    else:
                        content_value += "{}:{},".format(key, values[key].shape)
                n_str += name + "=" + "{" + content_value + "}"
            else:
                if isinstance(values, list):
                    content_value = f"[" + ", ".join(["{}".format(len(k)) for k in values]) + "]"
                    n_str += f"{name}={ {content_value} }, "
                else:
                    content_value = "{}".format(values.shape)
                    n_str += name + "=" + "" + content_value + ""
        n_str += "\n\t"

        _str = _str.replace("tensor(", n_str)
        return _str

    def get_slices(self, dim_values, label=None):
        """
        Get a list of slices for each named dimension.

        Example : for a tensor `t` with names ('T', 'C', 'H', 'W')
            t[t.get_slices({"C":1})] is equivalent to t[:,1,:,:]


        Parameters
        ----------
        dim_values : dict
            keys : name of the dimension
            value : slice or value to index on this dimension
            the dimensions that are not keys of the dict have a default value of slice(None)

        Returns
        -------
        slices : list of slice object
            This can be used
        """
        assert not any(
            key not in self.names for key in dim_values
        ), "One of the desired slice dim name is not in the current names of this augmented tensor."
        slices = []
        names = self.names if label is None else label.names
        for dim in names:
            sl = dim_values[dim] if dim in dim_values else slice(None)
            slices.append(sl)
        return slices

    def recursive_apply_on_children_(self, func):
        """
        Recursively apply function on labels to modify tensor inplace
        """

        def __apply(l):
            return func(l).recursive_apply_on_children_(func)

        for name in self._children_list:
            label = getattr(self, name)
            modified_label = self.apply_on_child(label, __apply)
            setattr(self, name, modified_label)
        return self

    def _hflip_label(self, label, **kwargs):
        """
        Returns label horizontally flipped if possible, else unmodified label.
        """
        try:
            label_flipped = label._hflip(**kwargs)
        except AttributeError:
            return label
        else:
            return label_flipped

    def hflip(self, **kwargs):
        """
        Flip AugmentedTensor horizontally, and its labels recursively

        Returns
        -------
        flipped : aloscene AugmentedTensor
            horizontally flipped tensor
        """

        flipped = self._hflip(**kwargs)
        flipped.recursive_apply_on_children_(lambda label: self._hflip_label(label, **kwargs))
        return flipped

    def _hflip(self, *args, **kwargs):
        raise Exception("This Augmented tensor should implement this method")

    def vflip(self, **kwargs):
        """
        Flip AugmentedTensor vertically, and its labels recursively

        Returns
        -------
        flipped : aloscene AugmentedTensor
            vertically flipped tensor
        """

        flipped = self._vflip(**kwargs)
        flipped.recursive_apply_on_children_(lambda label: self._vflip_label(label, **kwargs))
        return flipped

    def _vflip(self, *args, **kwargs):
        raise Exception("This Augmented tensor should implement this method")

    def resize(self, size, **kwargs):
        """
        Resize AugmentedTensor, and its labels recursively

        Parameters
        ----------
        size : tuple of int
            target size (H, W)

        Returns
        -------
        resized : aloscene AugmentedTensor
            resized tensor
        """
        h, w = size
        size01 = (h / self.H, w / self.W)

        def resize_func(label):
            # resize with relative coordinates if possible, else return unmodified label
            try:
                label_resized = label._resize(size01, **kwargs)
                return label_resized
            except AttributeError:
                return label

        resized = self._resize(size01, **kwargs)
        resized.recursive_apply_on_children_(resize_func)

        return resized

    def _resize(self, *args, **kwargs):
        raise Exception("This Augmented tensor should implement this method")

    def rotate(self, angle, **kwargs):
        """
        Rotate AugmentedTensor, and its labels recursively

        Parameters
        ----------
        angle : float

        Returns
        -------
        rotated : aloscene AugmentedTensor
            rotated tensor
        """

        def rotate_func(label):
            try:
                label_rotated = label._rotate(angle, **kwargs)
                return label_rotated
            except AttributeError:
                return label

        rotated = self._rotate(angle, **kwargs)
        rotated.recursive_apply_on_children_(rotate_func)

        return rotated

    def _crop_label(self, label, H_crop, W_crop, **kwargs):
        try:
            label_resized = label._crop(H_crop, W_crop, **kwargs)
            return label_resized
        except AttributeError:
            return label

    def crop(self, H_crop: tuple, W_crop: tuple, **kwargs):
        """
        Crop AugmentedTensor, and its labels recursively

        Parameters
        ----------
        H_crop: tuple
            (start, end) between 0 and 1
        W_crop: tuple
            (start, end) between 0 and 1

        Returns
        -------
        croped : aloscene AugmentedTensor
            croped tensor
        """
        if H_crop[0] < 0.0 or H_crop[1] > 1.0:
            raise Exception("H_crop is expected to be between 0 and 1 but found {}".format(H_crop))
        elif W_crop[0] < 0.0 or W_crop[1] > 1.0:
            raise Exception("W_crop is expected to be between 0 and 1 but found {}".format(W_crop))
        croped = self._crop(H_crop, W_crop, **kwargs)
        croped.recursive_apply_on_children_(lambda l: self._crop_label(l, H_crop, W_crop, **kwargs))

        return croped

    def _crop(self, *args, **kwargs):
        raise Exception("This Augmented tensor should implement this method")

    def _pad_label(self, label, offset_y, offset_x, **kwargs):
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
        padded.recursive_apply_on_children_(
            lambda label: self._pad_label(label, offset_y, offset_x, **kwargs)
        )
        return padded

    def _spatial_shift_label(self, label, shift_y, shift_x, **kwargs):
        try:
            label_shift = label._spatial_shift(shift_y, shift_x, **kwargs)
            return label_shift
        except AttributeError:
            return label

    def spatial_shift(self, shift_y: float, shift_x: float, **kwargs):
        """
        Spatially shift the AugmentedTensor and its labels recursively

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
        shifted = self._spatial_shift(shift_y, shift_x, **kwargs)
        shifted.recursive_apply_on_children_(
            lambda label: self._spatial_shift_label(label, shift_y, shift_x, **kwargs)
        )
        return shifted

    def _spatial_shift(self, shift_y, shift_x, **kwargs):
        raise Exception(f"This Augmented tensor {type(self)} should implement this method")

    def __getattribute__(self, name: str) -> Any:
        """Somehow hacky but this is maybe the only way to pass Augmented
        tensor through the datapipeline when doing multi-gpu training. Looking
        forward for the next pytorch release.

        Therefore, when passing your Augmented Tensor through a Datapipeline (the output of __getitem__)
        you must rename your augmented tensor using `augmented_tensor.rename(auto_restore=True)`.
        """
        if (
            name == "names"
            and self._auto_restore_names
            and self._saved_names is not None
            and any(self._saved_names)
            and len(self._saved_names) == len(self.shape)
        ):
            self._auto_restore_names = False
            self.reset_names()
        return super().__getattribute__(name)

    def _hflip(self, *args, **kwargs):
        # Must be implement by child class to handle hflip
        return self.clone()

    def _resize(self, *args, **kwargs):
        # Must be implement by child class to handle resize
        return self.clone()

    def _rotate(self, *args, **kwargs):
        # Must be implement by child class to handle rotate
        return self.clone()

    def _crop(self, *args, **kwargs):
        # Must be implement by child class to handle crop
        return self.clone()

    def _pad(self, *args, **kwargs):
        # Must be implement by child class to handle padding
        return self.clone()

    def get_view(self, *args, **kwargs):
        # Must be implement by child class to handle display
        pass
