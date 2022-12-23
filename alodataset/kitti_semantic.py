import os
import numpy as np
import cv2
import torch
from typing import Union

from alodataset import BaseDataset, Split, SplitMixin
from aloscene import Frame, Mask, Labels


class KittiSemanticDataset(BaseDataset, SplitMixin):
    SPLIT_FOLDERS = {Split.TRAIN: "training", Split.TEST: "testing"}

    def __init__(self, name="kitti_semantic", **kwargs):
        super().__init__(name=name, **kwargs)

        if self.sample:
            return

        self.split_folder = os.path.join(self.dataset_dir, self.get_split_folder())
        left_img_folder = os.path.join(self.split_folder, "image_2")
        n_samples = len([file for file in os.listdir(left_img_folder)])

        self.category = [
            ("unlabeled", 0),
            ("ego vehicle", 1),
            ("rectification border", 2),
            ("out of roi", 3),
            ("static", 4),
            ("dynamic", 5),
            ("ground", 6),
            ("road", 7),
            ("sidewalk", 8),
            ("parking", 9),
            ("rail track", 10),
            ("building", 11),
            ("wall", 12),
            ("fence", 13),
            ("guard rail", 14),
            ("bridge", 15),
            ("tunnel", 16),
            ("pole", 17),
            ("polegroup", 18),
            ("traffic light", 19),
            ("traffic sign", 20),
            ("vegetation", 21),
            ("terrain", 22),
            ("sky", 23),
            ("person", 24),
            ("rider", 25),
            ("car", 26),
            ("truck", 27),
            ("bus", 28),
            ("caravan", 29),
            ("trailer", 30),
            ("train", 31),
            ("motorcycle", 32),
            ("bicycle", 33),
            ("license plate", -1),
        ]

        self.items = {}
        for idx in range(n_samples):
            self.items[idx] = {
                "left": os.path.join(self.split_folder, f"image_2/{idx:06d}_10.png"),
                "instance": os.path.join(self.split_folder, f"instance/{idx:06d}_10.png")
                if self.split == Split.TRAIN
                else None,
                "semantic": os.path.join(self.split_folder, f"semantic/{idx:06d}_10.png")
                if self.split == Split.TRAIN
                else None,
            }

    def rgb2id(self, color: Union[list, np.ndarray]):
        if isinstance(color, np.ndarray) and len(color.shape) == 3:
            if color.dtype == np.uint8:
                color = color.astype(np.int32)
            return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
        return int(color[0] + 256 * color[1] + 256 * 256 * color[2])

    def getitem(self, idx) -> Frame:
        item = self.items[idx]

        if self.sample:
            return BaseDataset.__getitem__(self, idx)

        frame = Frame(item["left"])

        if item["instance"] is not None:
            instance = cv2.imread(item["instance"], cv2.IMREAD_UNCHANGED)
            instance = np.array(instance, dtype=np.int32)
            instance_gt = instance % 256
            instance_gt = instance_gt.astype(np.uint8)
            semantic_gt = instance // 256
            semantic_gt = semantic_gt.astype(np.uint8)

            masks = []
            labels_list = []

            # For each instance of each category
            for cat in self.category:
                for unique in np.unique(instance_gt[semantic_gt == cat[1]]):
                    # Check for the object we want and the instance we want
                    object = (semantic_gt == cat[1]) & (instance_gt == unique)
                    masks.append(Mask(np.expand_dims(object, axis=0), names=("N", "H", "W")))
                    # Append the category id on the list
                    labels_list.append(cat[1])

            labels = torch.as_tensor([label for label in labels_list], dtype=torch.float32)
            # Labels store the list of categories id and a list off all categories
            labels_2d = Labels(labels, labels_names=[cat[0] for cat in self.category], names=("N"), encoding="id")
            all_masks = torch.cat(masks, dim=0)
            frame.append_segmentation(Mask(all_masks, labels=labels_2d, names=("N", "H", "W")))

        return frame


if __name__ == "__main__":
    from random import randint

    dataset = KittiSemanticDataset(sample=True)
    obj = dataset.getitem(randint(0, len(dataset) - 1))
    obj.get_view().render()
