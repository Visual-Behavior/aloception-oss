from typing import Optional
import torch

from alodataset import Mot17, Split
from alonet.detr import CocoDetection2Detr
from aloscene.renderer import View

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

######
# Aloception Intro
######
#
# Aloception is developed under the pytorchlightning framework.
# For its use, aloception provides different modules that facilitate its use.
# For more information, see https://www.pytorchlightning.ai/


######
# Module customization (custom dataset)
######
#
# Aloception allows flexibility in custom modules creation.
# This tutorial explains how to load and manipulate the MOT17 dataset to detect people, using the CocoDetection2Detr module.
#
# All information available at:
#   [x] MOT17 dataset: https://motchallenge.net/data/MOT17/
#   [x] LightningDataModule: https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html


# MOT17 dataset implement as a LightningDataModule, based on CocoDetectio2Detr module
class Mot17DetectionDetr(CocoDetection2Detr):

    # Method overwrite
    def setup(self, stage: Optional[str] = None) -> None:
        self.coco_train = Mot17(  # Change default dataset to MOT17
            split=Split.TRAIN,
            sequence_size=1,
            mot_sequences=["MOT17-02-DPM", "MOT17-02-SDP"],
            transform_fn=self.train_transform,
        )
        self.coco_val = Mot17(  # Change default dataset to MOT17
            split=Split.TRAIN,
            sequence_size=1,
            mot_sequences=["MOT17-05-DPM", "MOT17-05-SDP"],
            transform_fn=self.val_transform,
        )


# Use of the new modules with their new functionalities
mot_loader = Mot17DetectionDetr()

for frames in mot_loader.val_dataloader():
    frames = frames[0].batch_list(frames)  # Remove temporal dim
    frames.get_view(frames.boxes2d,).render(View.CV)
