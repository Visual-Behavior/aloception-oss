# -*- coding: utf-8 -*-
"""Read a dataset in **COCO JSON** format and transform each element
in a :mod:`Frame object <aloscene.frame>`. Ideal for object detection applications.
"""

import os
import numpy as np
import torch

from alodataset import BaseDataset
from aloscene import BoundingBoxes2D, Frame, Labels, Mask
from collections import defaultdict
from pathlib import Path
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from typing import Dict, Union



class CocoBaseDataset(BaseDataset):
    """
    Attributes
    ----------
    label_names : list
        List of labels according to their corresponding positions
    prepare : :mod:`BaseDataset <base_dataset>`

    Parameters
    ----------
    img_folder : str
        Path to the image folder relative at :attr:`dataset_dir` (stored into the aloception config file)
    ann_file : str
        Path to the annotation file relative at :attr:`dataset_dir` (stored into the aloception config file)
    name : str, optional
        Key of database name in :attr:`alodataset_config.json` file, by default *coco*
    return_masks : bool, optional
        Include masks labels in the output, by default False
    classes : list, optional
        List of classes to be filtered in the annotation reading process, by default None
    fix_classes_len : int, optional
        Fix to a specific number the number of classes, filling the rest with "N/A" value.
        Use when the number of model outputs does not match with the number of classes in the dataset,
        by default None
    return_multiple_labels : bool, optional
        Return Labels as a dictionary, with all posible categories found in annotations file, by default False
    **kwargs : dict
        :mod:`BaseDataset <base_dataset>` optional parameters

    Raises
    ------
    Exception
        If a :attr:`classes` list is decided, each label must be inside of :attr:`label_names` list attribute
    ValueError
        If :attr:`fix_classes_len` is desired, :attr:`fix_classes_len` > len(:attr:`label_names`)

    Examples
    --------
        >>> coco_ds = CocoBaseDataset(
        ... img_folder = "val2017",
        ... ann_file = "annotations/instances_val2017.json",
        )
        >>> frames = next(iter(coco_ds.train_loader()))
        >>> frames = frames[0].batch_list(frames)
        >>> frames.get_view(frames.boxes2d,).render()
    """

    def __init__(
        self,
        img_folder: str = None,
        ann_file: str = None,
        name: str = "coco",
        return_masks=False,
        classes: list = None,
        fix_classes_len: int = None,
        return_multiple_labels: list = None,
        **kwargs,
    ):
        super(CocoBaseDataset, self).__init__(name=name, **kwargs)
        if self.sample:
            return
        else:
            assert img_folder is not None, "When sample = False, img_folder must be given."
            assert ann_file is not None or "test" in img_folder, "When sample = False and the test split is not used, ann_file must be given."


        # Create properties
        self.img_folder = os.path.join(self.dataset_dir, img_folder)

        if "test" in img_folder:
            #get a list of  indices that don't rely on the annotation file
            self.items = [int(Path(os.path.join(self.img_folder, f)).stem) for f in os.listdir(self.img_folder) if os.path.isfile(os.path.join(self.img_folder, f))]
            return

        self.coco = COCO(os.path.join(self.dataset_dir, ann_file))
        self.items = list(sorted(self.coco.imgs.keys()))

        # Setup the class names
        cats = self.coco.loadCats(self.coco.getCatIds())
        nb_category = max(cat["id"] for cat in cats)
        label_names = ["N/A"] * (nb_category + 1)
        for cat in cats:
            label_names[cat["id"]] = cat["name"]

        self._ids_renamed = classes
        self.label_names = label_names
        if classes is not None:
            notclass = [label for label in classes if label not in self.label_names]
            if len(notclass) > 0:  # Ignore all labels not in classes
                raise Exception(
                    f"The {notclass} classes dont match in label_names list. Possible values: {self.label_names}"
                )

            self.label_names = classes
            self._ids_renamed = [-1 if label not in classes else classes.index(label) for label in label_names]
            self._ids_renamed = np.array(self._ids_renamed)

            # Check each annotation and keep only that have at least 1 box in classes list
            ids = []
            for i in self.items:
                target = self.coco.loadAnns(self.coco.getAnnIds(i))
                if any([self._ids_renamed[bbox["category_id"]] >= 0 for bbox in target]):
                    ids.append(i)
            self.items = ids  # Remove images without bboxes with classes in classes list

        # Fix lenght of label_names to a desired `fix_classes_len`
        if fix_classes_len is not None:
            self._fix_classes(fix_classes_len)
        self.prepare = ConvertCocoPolysToMask(return_masks)

        # Process to return multiple labels : Create encoding objects
        self.label_types, self.label_types_names = None, None
        if return_multiple_labels:
            dict_cats = {
                lbl: self.coco.loadCats(self.coco.getCatIds(lbl))[0] if lbl != "N/A" else {}
                for lbl in self.label_names
            }
            self.label_types, self.label_types_names = self._get_label_types(dict_cats)

    def _get_label_types(self, dict_cats: Dict[dict, list]):
        label_types, label_types_names = defaultdict(list), defaultdict(list)

        # Get name list by each super-category
        for cat in dict_cats.values():
            for ltype, name in cat.items():
                if ltype in ["id", "name"]:
                    break
                label_types_names[ltype].append(name)

        # Remove duplicate types names and add N/A class
        for ltype in label_types_names:
            label_types_names[ltype] = sorted(list(set(label_types_names[ltype])))
            label_types_names[ltype].append("N/A")

        # Get encoding for each category
        for lbl in self.label_names:
            cat = dict_cats[lbl]
            for ltype in label_types_names:
                if len(cat) > 0:
                    label_types[ltype].append(label_types_names[ltype].index(cat[ltype]))
                else:
                    label_types[ltype].append(label_types_names[ltype].index("N/A"))

        return label_types, label_types_names

    def _fix_classes(self, new_label_size):
        if new_label_size > len(self.label_names):
            self.label_names += ["N/A"] * (new_label_size - len(self.label_names))
        else:
            raise ValueError(
                f"fix_classes_len must be higher than the lenght of label_names ({len(self.label_names)})."
            )

    def _append_labels(self, element: Union[BoundingBoxes2D, Mask], target):
        def append_new_labels(element, ltensor, lnames, name):
            label_2d = Labels(ltensor.to(torch.float32), labels_names=lnames, names=("N"), encoding="id")
            element.append_labels(label_2d, name=name)

        labels = target["labels"]
        # Append main labels
        if self.label_types is None:
            append_new_labels(element, labels, self.label_names, None)
            return
        else:
            append_new_labels(element, labels, self.label_names, "category")

        # Append supercategory labels
        for ktype in self.label_types:
            append_new_labels(
                element, torch.as_tensor(self.label_types[ktype])[labels], self.label_types_names[ktype], ktype
            )

        # Append specific labels
        append_new_labels(element, target["iscrowd"].to(torch.float32), None, "iscrowd")

    def _target2aloscene(self, target, frame):
        # Clean index by unique classes filtered
        if self._ids_renamed is not None:
            new_labels = target["labels"].numpy().astype(int)
            new_labels = self._ids_renamed[new_labels]

            # Keep only valid boxes
            idxs = np.where(new_labels >= 0)[0]
            target["boxes"] = target["boxes"][idxs]
            target["labels"] = torch.from_numpy(new_labels[idxs])

            if self.prepare.return_masks:
                target["masks"] = target["masks"][idxs]

            if self.label_types is not None:
                target["iscrowd"] = target["iscrowd"][idxs]

        # Create and append labels to boxes
        boxes = BoundingBoxes2D(
            target["boxes"], boxes_format="xyxy", absolute=True, frame_size=frame.HW, names=("N", None)
        )
        self._append_labels(boxes, target)

        # Create and append labels to masks
        segmentation = None
        if self.prepare.return_masks:
            segmentation = Mask(target["masks"], names=("N", "H", "W"))
            self._append_labels(segmentation, target)

        return boxes, segmentation

    def getitem(self, idx):
        """Get the :mod:`Frame <aloscene.frame>` corresponds to *idx* index

        Parameters
        ----------
        idx : int
            Index of the frame to be returned

        Returns
        -------
        :mod:`Frame <aloscene.frame>`
            Frame with their corresponding boxes and labels
        """
        if self.sample:
            return BaseDataset.__getitem__(self, idx)

        image_id = self.items[idx]
        if "test" in self.img_folder:
            #get the filename from image_id without relying on annotation file
            return Frame(os.path.join(self.img_folder, f"{str(image_id).zfill(12)}.jpg"))

        frame = Frame(os.path.join(self.img_folder, self.coco.loadImgs(image_id)[0]["file_name"]))

        target = self.coco.loadAnns(self.coco.getAnnIds(image_id))
        target = {"image_id": image_id, "annotations": target}
        _, target = self.prepare(frame, target)

        # Append target into frame
        boxes, segmentation = self._target2aloscene(target, frame)
        frame.append_boxes2d(boxes)
        frame.segmentation = segmentation
        return frame


class ConvertCocoPolysToMask(object):
    """Class to convert polygons or keypoints into boxes

    Attributes
    ----------
    return_masks : bool, optional
        Return in target the mask as attribute, by default False

    Examples
    --------
        >>> coco_mask = ConvertCocoPolysToMask()
        >>> new_image, new_target = coco_mask(image, target)
    """

    def __init__(self, return_masks: bool = False):
        self.return_masks = return_masks

    def convert_coco_poly_to_mask(self, segmentations, height, width):
        masks = []
        for polygons in segmentations:
            assert len(polygons) > 0, "Annotations file has not info about segmentation"
            if isinstance(polygons, list):
                polygons = coco_mask.frPyObjects(polygons, height, width)
            mask = coco_mask.decode(polygons)
            if len(mask.shape) < 3:
                mask = mask[..., None]
            mask = torch.as_tensor(mask, dtype=torch.uint8)
            mask = mask.any(dim=2)
            masks.append(mask)
        if masks:
            masks = torch.stack(masks, dim=0)
        else:
            masks = torch.zeros((0, height, width), dtype=torch.uint8)
        return masks

    def __call__(self, image, target):

        w, h = image.shape[-1], image.shape[-2]

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if "iscrowd" not in obj or obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = self.convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


if __name__ == "__main__":
    coco_dataset = CocoBaseDataset(sample=False, img_folder="test2017")
    #checking if regular getitem works
    frame = coco_dataset[0]
    frame.get_view().render()

    #check if dataloader works
    for f, frames in enumerate(coco_dataset.train_loader(batch_size=2)):
        frames = Frame.batch_list(frames)
        frames.get_view().render()
        if f > 1:
            break
