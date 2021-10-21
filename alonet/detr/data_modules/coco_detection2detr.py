from argparse import ArgumentParser, Namespace
from typing import Optional

from alonet.detr.data_modules import Data2Detr
import alodataset


class CocoDetection2Detr(Data2Detr):
    """LightningDataModule to use coco dataset in Detr models

    Attributes
    ----------
    label_names : list
        List of labels names use to encode the classes by index

    Parameters
    ----------
    classes : list, optional
        List to classes to be filtered in dataset, by default None
    name : str, optional
        Database name, by default "coco"
    train_folder : str, optional
        Image folder to train, by default "train2017"
    train_ann : str, optional
        Json annotation file to train, by default "annotations/instances_train2017.json"
    val_folder : str, optional
        Image folder to valid, by default "val2017"
    val_ann : str, optional
        Json annotation file to valid, by default "annotations/instances_val2017.json"
    return_masks : bool, optional
        For each frame return masks in segmentation attribute, by default False
    args : Namespace, optional
        Attributes stored in specific Namespace, by default None
    **kwargs
        :mod:`~alonet.detr.Data2Detr` additional parameters
    """

    def __init__(
        self,
        args: Namespace = None,
        name: str = "coco",
        classes: list = None,
        train_folder: str = "train2017",
        train_ann: str = "annotations/instances_train2017.json",
        val_folder: str = "val2017",
        val_ann: str = "annotations/instances_val2017.json",
        return_masks: bool = False,
        **kwargs
    ):
        # Update class attributes with args and kwargs inputs
        self.train_loader_kwargs = dict(
            img_folder=train_folder,
            ann_file=train_ann,
            # Split=Split.TRAIN if not self.train_on_val else Split.VAL,
            classes=classes,
            name=name,
            return_masks=return_masks,
        )
        self.val_loader_kwargs = dict(
            img_folder=val_folder,
            ann_file=val_ann,
            # split=Split.VAL,
            classes=classes,
            name=name,
            return_masks=return_masks,
        )

        super().__init__(args=args, **kwargs)

        if self.train_on_val:
            self.train_loader_kwargs["img_folder"] = val_folder
            self.train_loader_kwargs["ann_file"] = val_ann

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            # Setup train/val loaders
            self.train_dataset = alodataset.CocoDetectionDataset(
                transform_fn=self.train_transform, sample=self.sample, **self.train_loader_kwargs
            )
            self.sample = self.train_dataset.sample or self.sample  # Update sample if user prompt is given
            self.val_dataset = alodataset.CocoDetectionDataset(
                transform_fn=self.val_transform, sample=self.sample, **self.val_loader_kwargs
            )
            self.sample = self.val_dataset.sample or self.sample  # Update sample if user prompt is given
            self.label_names = self.val_dataset.label_names if hasattr(self.val_dataset, "label_names") else None


if __name__ == "__main__":
    # setup data
    loader_kwargs = dict(
        name="rabbits",
        train_folder="train",
        train_ann="train/_annotations.coco.json",
        val_folder="valid",
        val_ann="valid/_annotations.coco.json",
    )

    args = CocoDetection2Detr.add_argparse_args(ArgumentParser()).parse_args()  # Help provider
    coco = CocoDetection2Detr(args, **loader_kwargs)
    coco.prepare_data()
    coco.setup()

    samples = next(iter(coco.train_dataloader()))
    samples[0].get_view().render()

    samples = next(iter(coco.val_dataloader()))
    samples[0].get_view().render()
