# -*- coding: utf-8 -*-
"""Based on :mod:`CocoBaseDataset object <alodataset.coco_detection_dataset>`. Read a dataset in **COCO JSON**
format with additional options to get multiples labels and transform each element in a
:mod:`Frame object <aloscene.frame>`. Ideal for object detection and segmentation applications.
"""

import os
import torch
import numpy as np
from pycocotools.coco import COCO
from collections import defaultdict

from alodataset import BaseDataset, CocoBaseDataset, Split, SplitMixin
from aloscene import Frame


class CocoDetectionDataset(CocoBaseDataset, SplitMixin):
    """
    Attributes
    ----------
    label_names : list
        List of labels according to their corresponding positions
    prepare : :mod:`BaseDataset <base_dataset>`

    Parameters
    ----------
    name : str, optional
        Key of database name in `alodataset_config.json` file, by default *coco*
    split : :class:`~alodataset.base_dataset.Split`, optional
        Handle a specific dataset, by defautl :attr:`Split.TRAIN`
    return_masks : bool, optional
        Include masks labels in the output, by default False
    classes : list, optional
        List of classes to be filtered in the annotation reading process, by default None
    fix_classes_len : int, optional
        Fix to a specific number the number of classes, filling the rest with "N/A" value.
        Use when the number of model outputs does not match with the number of classes in the dataset,
        by default None
    include_stuff_cats : bool, optional
        Include :attr:`STUFF` objects in reading process, by default False
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
    Exception
        All categories in annotations files must have different ids

    Examples
    --------
        >>> coco_ds = CocoDetectionDataset()
        >>> frames = next(iter(coco_ds.train_loader()))
        >>> frames = frames[0].batch_list(frames)
        >>> frames.get_view(frames.boxes2d,).render()
    """

    SPLIT_FOLDERS = {Split.VAL: "val2017", Split.TRAIN: "train2017"}
    SPLIT_ANN_FILES = {
        Split.VAL: "annotations/instances_val2017.json",
        Split.TRAIN: "annotations/instances_train2017.json",
    }
    SPLIT_ANN_STUFF_FILES = {
        Split.VAL: "annotations/stuff_val2017.json",
        Split.TRAIN: "annotations/stuff_train2017.json",
    }

    def __init__(
        self,
        name: str = "coco",
        classes: list = None,
        include_stuff_cats: bool = False,
        fix_classes_len: int = None,
        split=Split.TRAIN,
        ann_file=None,
        **kwargs,
    ):
        SplitMixin.__init__(self, split)
        super(CocoDetectionDataset, self).__init__(
            name=name,
            img_folder=self.get_split_folder(),
            ann_file=ann_file or self.get_split_ann_file(),
            classes=None if include_stuff_cats else classes,
            fix_classes_len=fix_classes_len,
            **kwargs,
        )

        # Extend annotations
        self.coco_stuff = None
        if include_stuff_cats:
            self.coco_stuff = COCO(os.path.join(self.dataset_dir, self.get_split_stuff_ann_file()))

            # Merge stuff labels into things labels
            cats = self.coco_stuff.loadCats(self.coco_stuff.getCatIds())
            nb_category = max(cat["id"] for cat in cats)
            label_names = ["N/A"] * (nb_category + 1)
            for cat in cats:
                label_names[cat["id"]] = cat["name"]
            for lid in range(max(len(self.label_names), len(label_names))):
                if lid < len(label_names):
                    if lid < len(self.label_names):
                        if self.label_names[lid] != "N/A" and label_names[lid] != "N/A":
                            raise Exception(
                                f"Two categories were found for id {lid}: "
                                + f"{self.label_names[lid]} (things) and {label_names[lid]} (stuffs). Ambiguity"
                            )
                        elif label_names[lid] != "N/A":
                            self.label_names[lid] = label_names[lid]
                    else:
                        self.label_names.append(label_names[lid])
                else:
                    break

            # Classes filter if desired
            if classes is not None:
                notclass = [label for label in classes if label not in self.label_names]
                if len(notclass) > 0:  # Ignore all labels not in classes
                    raise Exception(
                        f"The {notclass} classes dont match in label_names list. Possible values: {self.label_names}"
                    )

                self._ids_renamed = [-1 if lbl not in classes else classes.index(lbl) for lbl in self.label_names]
                self._ids_renamed = np.array(self._ids_renamed)
                self.label_names = classes

                # Check each annotation and keep only that have at least 1 box in classes list
                ids = []
                for i in self.items:
                    # Only take into account images with things annotations
                    target = self.coco.loadAnns(
                        self.coco.getAnnIds(i)
                    )  # + self.coco_stuff.loadAnns(self.coco_stuff.getAnnIds(i))
                    if any([self._ids_renamed[bbox["category_id"]] >= 0 for bbox in target]):
                        ids.append(i)
                self.items = ids  # Remove images without bboxes with classes in classes list

                if fix_classes_len is not None:
                    self._fix_classes(fix_classes_len)

        # Re-calcule encoding label types (+stuff)
        if self.label_types is not None:
            dict_cats = dict()
            self.label_types = defaultdict(list)
            self.label_types_names = dict(isthing=["stuff", "thing", "N/A"])

            for lbl in self.label_names:
                cat = self.coco.loadCats(self.coco.getCatIds(lbl))
                isthing = 1
                if len(cat) == 0 and include_stuff_cats:
                    cat = self.coco_stuff.loadCats(self.coco_stuff.getCatIds(lbl))
                    isthing = 0 if len(cat) > 0 else 2
                dict_cats[lbl] = cat[0] if len(cat) > 0 else {}
                self.label_types["isthing"].append(isthing)
            label_types, label_types_names = self._get_label_types(dict_cats)
            self.label_types.update(label_types)
            self.label_types_names.update(label_types_names)

    def get_split_ann_file(self):
        """Get annotation file according to :attr:`split`.

        Returns
        -------
        str
            annotation file
        """
        assert self.split in self.SPLIT_ANN_FILES
        return self.SPLIT_ANN_FILES[self.split]

    def get_split_stuff_ann_file(self):
        """Get stuff annotation file according to :attr:`split`.

        Returns
        -------
        str
            annotation file
        """
        assert self.split in self.SPLIT_ANN_STUFF_FILES
        return self.SPLIT_ANN_STUFF_FILES[self.split]

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

        frame = super().getitem(idx)
        if self.coco_stuff is not None:
            image_id = self.items[idx]
            target = self.coco_stuff.loadAnns(self.coco_stuff.getAnnIds(image_id))
            target = {"image_id": image_id, "annotations": target}
            _, target = self.prepare(frame, target)

            boxes, segmentation = self._target2aloscene(target, frame)
            frame.boxes2d = torch.cat([frame.boxes2d, boxes], dim=0)
            if self.prepare.return_masks:
                frame.segmentation = torch.cat([frame.segmentation, segmentation], dim=0)

        return frame


if __name__ == "__main__":
    coco_dataset = CocoDetectionDataset(split=Split.VAL, return_multiple_labels=True)
    for f, frames in enumerate(coco_dataset.stream_loader(num_workers=1)):
        frames = Frame.batch_list([frames])
        frames.get_view().render()
        if f > 1:
            break
