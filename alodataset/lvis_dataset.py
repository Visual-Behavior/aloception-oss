# -*- coding: utf-8 -*-
"""Read a dataset in **COCO JSON** format and transform each element
in a :mod:`Frame object <aloscene.frame>`. Ideal for object detection applications.
"""

import torch
import torch.utils.data
from pycocotools import mask as coco_mask
import numpy as np
import os
import random

# import detr.datasets.transforms as T

from torchvision.datasets.coco import CocoDetection
from PIL import Image
from alodataset import BaseDataset, CocoDetectionDataset
from aloscene import BoundingBoxes2D, Frame, Labels, Mask



class LvisDataset(CocoDetectionDataset):
    """ Lvis Dataset https://www.lvisdataset.org/.
    Same usage as CocoDetectionDataset. However, the
    `img_folder` does not need to be passed to the model. If the parameters
    is passed, its gonna be ignore.
    The dataset_dir is therefore supposed to contains at the root, the `val2017` folder and
    the `train2017` folder.
    """


    def __init__(self, **kwargs):
        CocoDetectionDataset.__init__(self, **kwargs)

        filtered_item = []

        for item in self.items:
            img_id = item
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            path = "/".join(self.coco.loadImgs(img_id)[0]['coco_url'].split("/")[-2:])
            if "train2017" not in path:
             filtered_item.append(item)
        assert type(self.items) == type(filtered_item)
        self.items = filtered_item



    def load_anns(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is a list of captions for the image.
        """
        coco = self.coco
        img_id = self.items[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        target = [ann for ann in anns]

        path = "/".join(coco.loadImgs(img_id)[0]['coco_url'].split("/")[-2:])

        img_path = os.path.join(self.dataset_dir, path)

        img = np.array(Image.open(img_path).convert('RGB'))

        return img, target


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
        #img, target = CocoDetectionDataset.__getitem__(self, idx)
        img, target = self.load_anns(idx)

        image_id = self.items[idx]
        target = {"image_id": image_id, "annotations": target}

        frame = Frame(np.transpose(np.array(img), [2, 0, 1]), names=("C", "H", "W"))

        _, target = self.prepare(frame, target)

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

def main():

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)


    """Main"""
    coco_dataset = LvisDataset(
        ann_file = "annotations/lvis_v1_val.json",
        return_masks=True
    )
    for f, frames in enumerate(coco_dataset.train_loader(batch_size=2)):
        frames = Frame.batch_list(frames)
        frames.get_view().render()


if __name__ == "__main__":
    main()