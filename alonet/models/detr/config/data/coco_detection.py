from dataclasses import dataclass, field

from alonet.common import BaseDataModuleConfig


@dataclass
class CocoDetectionConfig(BaseDataModuleConfig):
    name: str = field(default="coco", metadata={"help": "Name of the dataset"})
    classes: list = field(default=None, metadata={"help": "List of classes to use"})
    train_folder: str = field(default="train2017", metadata={"help": "Path to the training folder"})
    train_ann: str = field(
        default="annotations/instances_train2017.json", metadata={"help": "Path to the training annotation file"}
    )
    val_folder: str = field(default="val2017", metadata={"help": "Path to the validation folder"})
    val_ann: str = field(
        default="annotations/instances_val2017.json", metadata={"help": "Path to the validation annotation file"}
    )
    return_masks: bool = field(default=False, metadata={"help": "Return masks"})
    train_on_val: bool = field(default=False, metadata={"help": "Train on validation set"})
