# -*- coding: utf-8 -*-
"""Read a dataset in **COCO JSON** format and transform each element
in a :mod:`Frame object <aloscene.frame>`. Ideal for object detection applications.
"""

import torch
import torch.utils.data
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
import numpy as np
import os

# import detr.datasets.transforms as T

from torchvision.datasets.coco import CocoDetection

from alodataset import BaseDataset
from aloscene import BoundingBoxes2D, Frame, Labels, Mask


class CocoDetectionSample(CocoDetection):
    def __init__(self, *args, **kwargs):
        if self.sample:
            return
        super().__init__(*args, **kwargs)


class CocoDetectionDataset(BaseDataset, CocoDetectionSample):
    def __init__(
        self,
        img_folder: str = None,
        ann_file: str = None,
        name: str = "coco",
        return_masks=False,
        classes: list = None,
        stuff_ann_file: str = None,
        **kwargs,
    ):
        """
        Attributes
        ----------
        CATEGORIES : set
            List of all unique tags read from the database
        labels_names : list
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
        stuff_ann_file: str, optional
            Additional annotations with new classes, by default None
        **kwargs : dict
            :mod:`BaseDataset <base_dataset>` optional parameters

        Raises
        ------
        Exception
            If a classes list is decided, each label must be inside of :attr:`CATEGORIES` list attribute

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

        if "sample" not in kwargs:
            kwargs["sample"] = False

        self.sample = kwargs["sample"]
        if not self.sample:
            assert img_folder is not None, "When sample = False, img_folder must be given."
            assert ann_file is not None, "When sample = False, ann_file must be given."

            self.name = name
            dataset_dir = BaseDataset.get_dataset_dir(self)
            img_folder = os.path.join(dataset_dir, img_folder)
            ann_file = os.path.join(dataset_dir, ann_file)
            stuff_ann_file = None if stuff_ann_file is None else os.path.join(dataset_dir, stuff_ann_file)
            kwargs["sample"] = self.sample

        super(CocoDetectionDataset, self).__init__(name=name, root=img_folder, annFile=ann_file, **kwargs)
        if self.sample:
            return

        cats = self.coco.loadCats(self.coco.getCatIds())
        self.coco_stuff = None
        if stuff_ann_file is not None:
            self.coco_stuff = COCO(stuff_ann_file)
            cats += self.coco_stuff.loadCats(self.coco_stuff.getCatIds())

        # Setup the class names
        nb_category = max(cat["id"] for cat in cats)
        if stuff_ann_file is not None and name == "coco":
            nb_category = 250  # From original repo https://github.com/facebookresearch/detr/blob/main/models/detr.py
        self.CATEGORIES = set([cat["name"] for cat in cats])
        labels_names = ["N/A"] * (nb_category + 1)
        for cat in cats:
            labels_names[cat["id"]] = cat["name"]

        self._ids_renamed = classes
        if classes is None:
            self.labels_names = labels_names
        else:
            notclass = [label for label in classes if label not in self.CATEGORIES]
            if len(notclass) > 0:  # Ignore all labels not in classes
                raise Exception(
                    f"The {notclass} classes dont match in CATEGORIES list. Possible values: {self.CATEGORIES}"
                )

            self.labels_names = classes
            self._ids_renamed = [-1 if label not in classes else classes.index(label) for label in labels_names]
            self._ids_renamed = np.array(self._ids_renamed)

            # Check each annotation and keep only that have at least 1 box in classes list
            ids = []
            for i in self.ids:
                target = CocoDetectionSample._load_target(self, i)
                if any([self.ids_renamed[bbox["category_id"]] >= 0 for bbox in target]):
                    ids.append(i)
            self.ids = ids  # Remove images without bboxes with classes in classes list

        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.items = self.ids

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
        img, target = CocoDetectionSample.__getitem__(self, idx)

        image_id = self.ids[idx]
        if self.coco_stuff is not None:
            target += self.coco_stuff.loadAnns(self.coco_stuff.getAnnIds(image_id))
        target = {"image_id": image_id, "annotations": target}

        frame = Frame(np.transpose(np.array(img), [2, 0, 1]), names=("C", "H", "W"))
        _, target = self.prepare(img, target)

        # Clean index by unique classes filtered
        if self._ids_renamed is not None:
            new_labels = target["labels"].numpy().astype(int)
            new_labels = self._ids_renamed[new_labels]

            # Keep only valid boxes
            idxs = np.where(new_labels >= 0)[0]
            target["boxes"] = target["boxes"][idxs]
            target["labels"] = torch.from_numpy(new_labels[idxs])

        labels_2d = Labels(
            target["labels"].to(torch.float32), labels_names=self.labels_names, names=("N"), encoding="id"
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

        w, h = image.size[0], image.size[1]

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


def show_random_frame(coco_loader):
    frames = next(iter(coco_loader.train_loader(batch_size=2)))
    frames = Frame.batch_list(frames)
    frames.get_view().render()


def main():
    """Main"""
    coco_dataset = CocoDetectionDataset(sample=True)
    for f, frames in enumerate(coco_dataset.train_loader(batch_size=2)):
        frames = Frame.batch_list(frames)
        frames.get_view().render()
        if f > 1:
            break


if __name__ == "__main__":
    main()
