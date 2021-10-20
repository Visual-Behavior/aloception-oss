# -*- coding: utf-8 -*-
"""Read a dataset in **COCO JSON** format and transform each element
in a :mod:`Frame object <aloscene.frame>`. Ideal for object detection applications.
"""

import torch
import torch.utils.data
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
import numpy as np
import os

from alodataset import BaseDataset
from aloscene import BoundingBoxes2D, Frame, Labels, Mask


class CocoDetectionDataset(BaseDataset):
    """
    Attributes
    ----------
    label_names : list
        List of labels according to their corresponding positions
    prepare : :mod:`BaseDataset <base_dataset>`

    Parameters
    ----------
    img_folder : str
        Path to the image folder relative at `dataset_dir` (stored into the aloception config file)
    ann_file : str
        Path to the annotation file relative at `dataset_dir` (stored into the aloception config file)
    name : str, optional
        Key of database name in `alodataset_config.json` file, by default *coco*
    return_masks : bool, optional
        Include masks labels in the output, by default False
    classes : list, optional
        List of classes to be filtered in the annotation reading process, by default None
    fix_classes_len : int, optional
        Fix to a specific number the number of classes, filling the rest with "N/A" value.
        Use when the number of model outputs does not match with the number of classes in the dataset,
        by default None
    **kwargs : dict
        :mod:`BaseDataset <base_dataset>` optional parameters

    Raises
    ------
    Exception
        If a classes list is decided, each label must be inside of :attr:`label_names` list attribute
    ValueError
        If fix_classes_len is desired, fix_classes_len > len(label_names)

    Examples
    --------
        >>> coco_ds = CocoDetectionDataset(
        ... img_folder = "val2017",
        ... ann_file = "annotations/instances_val2017.json",
        ... mode = "validation"
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
        **kwargs,
    ):
        super(CocoDetectionDataset, self).__init__(name=name, **kwargs)
        if self.sample:
            return
        else:
            assert img_folder is not None, "When sample = False, img_folder must be given."
            assert ann_file is not None, "When sample = False, ann_file must be given."

        # Create properties
        self.img_folder = os.path.join(self.dataset_dir, img_folder)
        self.coco = COCO(os.path.join(self.dataset_dir, ann_file))
        self.return_masks = return_masks
        self.items = list(sorted(self.coco.imgs.keys()))

        # Setup the class names
        cats = self.coco.loadCats(self.coco.getCatIds())
        nb_category = max(cat["id"] for cat in cats)
        label_names = ["N/A"] * (nb_category + 1)
        for cat in cats:
            label_names[cat["id"]] = cat["name"]

        self._ids_renamed = classes
        if classes is None:
            self.label_names = label_names
        else:
            notclass = [label for label in classes if label not in label_names]
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
            if fix_classes_len > len(self.label_names):
                self.label_names += ["N/A"] * (fix_classes_len - len(self.label_names))
            else:
                raise ValueError(
                    f"fix_classes_len must be higher than the lenght of label_names ({len(self.label_names)})."
                )
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def getitem(self, idx):
        """Get the :mod:`Frame <aloscene.frame>` corresponds to *idx* index

        Parameters
        ----------
        idx : int
            Index of the frame to be returned

        Returns
        -------
        aloscene.Frame
            Frame with their corresponding boxes and labels
        """
        if self.sample:
            return BaseDataset.__getitem__(self, idx)

        image_id = self.items[idx]
        frame = Frame(os.path.join(self.img_folder, self.coco.loadImgs(id)[0]["file_name"]))
        target = self.coco.loadAnns(self.coco.getAnnIds(id))
        target = {"image_id": image_id, "annotations": target}
        _, target = self.prepare(frame, target)

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

        labels_2d = Labels(
            target["labels"].to(torch.float32), labels_names=self.label_names, names=("N"), encoding="id"
        )
        boxes = BoundingBoxes2D(
            target["boxes"],
            boxes_format="xyxy",
            absolute=True,
            frame_size=frame.HW,
            names=("N", None),
            labels=labels_2d,
        )
        frame.append_boxes2d(boxes)

        if self.prepare.return_masks:
            segmentation = Mask(target["masks"], names=("N", "H", "W"), labels=labels_2d)
            frame.append_segmentation(segmentation)

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
    coco_dataset = CocoDetectionDataset(sample=True)
    for f, frames in enumerate(coco_dataset.train_loader(batch_size=2)):
        frames = Frame.batch_list(frames)
        frames.get_view().render()
        if f > 1:
            break
