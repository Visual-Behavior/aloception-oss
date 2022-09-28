import torch
import numpy as np
from typing import Dict, List
from aloscene import Frame, Mask
from alodataset import Split, SplitMixin
from alodataset.kitti_depth import KittiBaseDataset


class KittiSplitDataset(KittiBaseDataset, SplitMixin):
    SPLIT_FOLDERS = {Split.VAL: "val", Split.TRAIN: "train"}

    def __init__(
        self,
        split=Split.TRAIN,
        add_depth_mask: bool = True,
        custom_drives: Dict[str, List[str]] = None,
        main_folder: str = "image_02",
        name: str = "kitti",
        **kwargs
    ):

        self.split = split
        self.add_depth_mask = add_depth_mask
        super().__init__(
            subsets=self.get_split_folder(),
            name=name,
            custom_drives=custom_drives,
            main_folder=main_folder,
            return_depth=True,
            **kwargs
        )

    def getitem(self, idx):
        frame = super().getitem(idx)
        if self.add_depth_mask:
            mask = Mask((frame.depth.as_tensor() != 0).type(torch.float), names=frame.depth.names)
            frame.depth.add_child("valid_mask", mask, align_dim=["B", "T"], mergeable=True)
        return frame
