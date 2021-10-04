"""Read a dataset with panoptic images annotations and transform each element
in a :mod:`Frame object <aloscene.frame>`. Ideal for object detection/segmentation applications.
"""
# Modify Panoptic dataset from original repository:
# https://github.com/facebookresearch/detr/blob/main/datasets/coco_panoptic.py

import os
import json
import numpy as np
import torch
from PIL import Image

from alodataset.utils.panoptic_utils import rgb2id
from alodataset.utils.panoptic_utils import masks_to_boxes

from alodataset import BaseDataset, SplitMixin, Split
from aloscene import Frame, BoundingBoxes2D, Mask, Labels


class CocoPanopticDataset(BaseDataset, SplitMixin):
    """Dataset uses in objects detection tasks. Import panoptic annotations from
    `2017 Panoptic Train/Val annotations <https://cocodataset.org/#download>`_ . Three elements are required:
    `image_folder`, `annotation_folder` and `annotation_file`. See
    `COCO 2018 Panoptic Segmentation Task <https://cocodataset.org/#panoptic-2018>`_ for more information.

    Attributes
    ----------
    SPLIT_FOLDERS: dict
        Contains the image folders for train/val/test
    SPLIT_ANN_FOLDERS:
        Contains the annotation folders for train/val/test
    SPLIT_ANN_FILES:
        Contains the annotation files for train/val/test
    labels_names : list
        List of labels according to their corresponding positions

    Parameters
    ----------
    name : str, optional
        Key of database name in `alodataset_config.json` file, by default *coco*
    split : alodataset.Split item, optional
        Define image folder and annotation file/folder to use, by default Split.TRAIN
    return_masks : bool, optional
        Include masks labels in the output, by default True
    classes : list, optional
        List of classes to be filtered in the annotation reading process, by default None
    fix_classes_len : int, optional
        Fix to a specific number the number of classes, filling the rest with "N/A" value.
        Use when the number of model outputs does not match with the number of classes in the dataset, by default 250
    **kwargs : dict
        :mod:`BaseDataset <base_dataset>` optional parameters

    Raises
    ------
    Exception
        :attr:`classes` attribute is not support when :attr:`label_names` does not exist in annotations file. Also,
        each element in :attr:`classes` attribute must be one of :attr:`label_names`.
    """

    SPLIT_FOLDERS = {Split.VAL: "val2017", Split.TRAIN: "train2017"}
    SPLIT_ANN_FOLDERS = {Split.VAL: "annotations/panoptic_val2017", Split.TRAIN: "annotations/panoptic_train2017"}
    SPLIT_ANN_FILES = {
        Split.VAL: "annotations/panoptic_val2017.json",
        Split.TRAIN: "annotations/panoptic_train2017.json",
    }

    def __init__(
        self,
        name: str = "coco",
        split=Split.TRAIN,
        return_masks: bool = True,
        classes: list = None,
        fix_classes_len: int = 250,  # Match with pre-trained weights
        **kwargs,
    ):
        super(CocoPanopticDataset, self).__init__(name=name, split=split, **kwargs)
        if self.sample:
            return

        # Create properties
        self.img_folder = os.path.join(self.dataset_dir, self.get_split_folder())
        self.ann_folder = os.path.join(self.dataset_dir, self.get_split_ann_folder())
        self.ann_file = os.path.join(self.dataset_dir, self.get_split_ann_file())
        self.return_masks = return_masks
        self.label_names = None
        self.items = self._get_sequences()

        # Fix classes if it is desired
        self._ids_renamed = classes
        if classes is not None:
            if self.label_names is None:
                raise Exception(
                    "'classes' attribute not support in datasets without 'categories' as attribute in annotation file"
                )
            notclass = [label for label in classes if label not in self.label_names]
            if len(notclass) > 0:  # Ignore all labels not in classes
                raise Exception(
                    f"The {notclass} classes dont match in CATEGORIES list. Possible values: {self.label_names}"
                )

            self._ids_renamed = [-1 if label not in classes else classes.index(label) for label in self.label_names]
            self._ids_renamed = np.array(self._ids_renamed)
            self.label_names = classes

            # Check each annotation and keep only that have at least 1 element in classes list
            items = []
            for i, (_, _, ann_info) in enumerate(self.items):
                target = ann_info["segments_info"]
                if any([self._ids_renamed[seg["category_id"]] >= 0 for seg in target]):
                    items.append(self.items[i])
            self.items = items

        # Fix number of label names if desired
        if fix_classes_len is not None:
            if fix_classes_len > len(self.label_names):
                self.label_names += ["N/A"] * (fix_classes_len - len(self.label_names))
            else:
                raise ValueError(
                    f"fix_classes_len must be higher than the lenght of label_names ({len(self.label_names)})."
                )

    def _get_sequences(self):
        """ """
        print("Loading annotations into memory...")
        # Read and aling images with annotations
        with open(self.ann_file, "r") as f:
            coco = json.load(f)
        coco["images"] = sorted(coco["images"], key=lambda x: x["id"])

        # Sanity check and items generation
        assert "annotations" in coco, "annotations must be provided"

        items = []
        for img, ann in zip(coco["images"], coco["annotations"]):
            assert img["file_name"][:-4] == ann["file_name"][:-4]
            assert "segments_info" in ann, "All annotation must have segments_info attribute"
            items.append(
                (
                    os.path.join(self.img_folder, ann["file_name"].replace(".png", ".jpg")),
                    os.path.join(self.ann_folder, ann["file_name"]),
                    ann,
                )
            )

        # Create annotations
        if "categories" in coco:
            nb_category = max(cat["id"] for cat in coco["categories"])
            self.label_names = ["N/A"] * (nb_category + 1)
            for cat in coco["categories"]:
                self.label_names[cat["id"]] = cat["name"]
        print("Done")
        return items

    def get_split_ann_folder(self):
        """Get annotation folder according to :attr:`split`.

        Returns
        -------
        str
            annotation folder
        """
        assert self.split in self.SPLIT_ANN_FOLDERS
        return self.SPLIT_ANN_FOLDERS[self.split]

    def get_split_ann_file(self):
        """Get annotation file according to :attr:`split`.

        Returns
        -------
        str
            annotation file
        """
        assert self.split in self.SPLIT_ANN_FILES
        return self.SPLIT_ANN_FILES[self.split]

    def getitem(self, idx):
        """Get the :mod:`Frame <aloscene.frame>` corresponds to *idx* index

        Parameters
        ----------
        idx : int
            Index of the frame to be returned

        Returns
        -------
        :mod:`Frame <aloscene.frame>`
            Frame with their corresponding boxes and masks attributes
        """
        if self.sample:
            return BaseDataset.__getitem__(self, idx)

        # Get elements
        img_path, ann_path, ann_info = self.items[idx]

        # Read mask annotation from panoptic image
        masks = np.asarray(Image.open(ann_path), dtype=np.uint32)
        masks = rgb2id(masks)  # Convert RGB to classesID

        ids = np.array([ann["id"] for ann in ann_info["segments_info"]])
        masks = masks == ids[:, None, None]
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        labels = torch.as_tensor([ann["category_id"] for ann in ann_info["segments_info"]], dtype=torch.int64)

        # Clean index by unique classes filtered
        if self._ids_renamed is not None:
            new_labels = self._ids_renamed[labels.numpy()]

            # Keep only valid masks
            idxs = np.where(new_labels >= 0)[0]
            masks = masks[idxs]
            labels = torch.from_numpy(new_labels[idxs])

        # Make aloscene.frame
        frame = Frame(img_path)

        labels_2d = Labels(labels.to(torch.float32), labels_names=self.label_names, names=("N"), encoding="id")
        boxes_2d = BoundingBoxes2D(
            masks_to_boxes(masks),
            boxes_format="xyxy",
            absolute=True,
            frame_size=frame.HW,
            names=("N", None),
            labels=labels_2d,
        )
        frame.append_boxes2d(boxes_2d)

        if self.return_masks:
            masks_2d = Mask(masks, names=("N", "H", "W"), labels=labels_2d)
            frame.append_segmentation(masks_2d)
        return frame


if __name__ == "__main__":
    coco_seg = CocoPanopticDataset(sample=True)
    for f, frames in enumerate(coco_seg.train_loader(batch_size=2)):
        frames = Frame.batch_list(frames)
        frames.get_view().render()
        if f > 1:
            break
