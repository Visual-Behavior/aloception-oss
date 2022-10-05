from argparse import ArgumentParser
import torch
from pytorch_lightning.callbacks import EarlyStopping

from alodataset import Mot17, Split
from alonet import common
from alonet.detr import CocoDetection2Detr, LitDetr, DetrR50Finetune
from alonet.detr import DetrObjectDetectorCallback
from alonet.callbacks import MetricsCallback, ApMetricsCallback
from aloscene import Labels
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
# This tutorial explains how to handled CocoDetection2Detr and LitDetrR50 modules
# to train a model that detects people in a MOT17 database.
#
# All information available at:
#   [x] MOT17 dataset: https://motchallenge.net/data/MOT17/
#   [x] LightningDataModule: https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
#   [x] LightningModule: https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html


# MOT17 dataset implement as a LightningDataModule, based on CocoDetectio2Detr module
class Mot17DetectionDetr(CocoDetection2Detr):

    # New function
    # Change id to the corresponding person id
    def _obj2person(self, frame):
        total_obj = frame.boxes2d.size(0)
        frame.boxes2d.labels = Labels(
            torch.zeros(total_obj).to(torch.float32), labels_names=["person"], names=("N"), encoding="id"
        )
        return frame

    # Method overwrite
    def train_transform(self, frame, **kwargs):
        frame = super().train_transform(frame[0], **kwargs)  # Remove temporal dim
        return self._obj2person(frame)

    def val_transform(self, frame, **kwargs):
        frame = super().val_transform(frame[0], **kwargs)  # Remove temporal dim
        return self._obj2person(frame)

    def setup(self, stage=None) -> None:
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


# LitDetr customization example
class LitDetrR50(LitDetr):
    # Method overwrite
    def __init__(self, args=None, **kwargs):
        # Prepare DetrR50Finetune, according to the training dataset
        model = DetrR50Finetune(num_classes=1, weights="detr-r50")
        super().__init__(args, model=model, **kwargs)
        self.model = self.model.to(device)

    def callbacks(self, data_loader):
        obj_detection_callback = DetrObjectDetectorCallback(val_frames=next(iter(data_loader.val_dataloader())))
        metrics_callback = MetricsCallback()
        ap_metrics_callback = ApMetricsCallback()
        early_callback = EarlyStopping(monitor="val_loss", patience=2)
        return [obj_detection_callback, metrics_callback, ap_metrics_callback, early_callback]


# Use of the new modules with their new functionalities to train model
parser = ArgumentParser(conflict_handler="resolve")
parser = Mot17DetectionDetr.add_argparse_args(parser)
parser = LitDetrR50.add_argparse_args(parser)
parser = common.add_argparse_args(parser)
args = parser.parse_args()

# Modules definition
mot_loader = Mot17DetectionDetr(args)
lit_detr = LitDetrR50(args)

# Train process
args.max_epochs = 10  # Fast training (pytorchlightning Trainer parameter)
lit_detr.run_train(
    data_loader=mot_loader,
    args=args,
    project="detr_mot",
)
lit_detr.model = lit_detr.model.eval().to(device)

# Check a random result
frame = next(iter(mot_loader.val_dataloader()))
frame = frame[0].batch_list(frame).to(device)
pred_boxes = lit_detr.inference(lit_detr(frame))[0]  # Inference from forward result
gt_boxes = frame[0].boxes2d

frame.get_view(
    [
        gt_boxes.get_view(frame[0], title="Ground truth boxes"),
        pred_boxes.get_view(frame[0], title="Predicted boxes"),
    ]
).render()
