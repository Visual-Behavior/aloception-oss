from argparse import ArgumentParser, Namespace
from typing import Optional

# from torch.utils.data import DataLoader

from alodataset import transforms as T  # , Split
import pytorch_lightning as pl
import alodataset
import alonet


class CocoDetection2Detr(pl.LightningDataModule):
    def __init__(
        self,
        args: Namespace = None,
        name: str = "coco",
        classes: list = None,
        train_folder: str = "train2017",
        train_ann: str = "annotations/instances_train2017.json",
        val_folder: str = "val2017",
        val_ann: str = "annotations/instances_val2017.json",
        **kwargs
    ):
        """LightningDataModule to use coco dataset in Detr models

        Parameters
        ----------
        batch_size : int, optional
            Batch size to use
        train_on_val : bool, optional
            Use train on validation
        num_workers : int, optional
            Nummer of workers to use
        no_augmentation : bool, optional
            Do not use augmentation to train the model
        size : tuple, optional
            If no augmentation (--no_augmentation) is used, --size can be used to resize all the frame.
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
        sample : bool, optional
            Use Sample instead of all dataset, by default False
        args : Namespace, optional
            Attributes stored in specific Namespace, by default None

        Notes
        -----
        Arguments entered by the user (kwargs) will replace those stored in args attribute
        """
        # Update class attributes with args and kwargs inputs

        super().__init__()
        alonet.common.pl_helpers.params_update(self, args, kwargs)

        self.size = list(self.size)
        self.size[0] = self.size[0] if self.size[0] is None else int(self.size[0])
        self.size[1] = self.size[1] if self.size[1] is None else int(self.size[1])

        self.train_loader_kwargs = dict(
            img_folder=train_folder if not self.train_on_val else val_folder,
            ann_file=train_ann if not self.train_on_val else val_ann,
            # Split=Split.TRAIN if not self.train_on_val else Split.VAL,
            classes=classes,
            name=name,
        )
        self.val_loader_kwargs = dict(
            img_folder=val_folder,
            ann_file=val_ann,
            # split=Split.VAL,
            classes=classes,
            name=name,
        )
        self.args = args
        self.val_check()  # Check val loader and set some previous parameters

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("DataModule")
        parser.add_argument("--batch_size", type=int, default=2, help="Batch size to use (default %(default)s)")
        parser.add_argument("--train_on_val", action="store_true")
        parser.add_argument(
            "--num_workers",
            type=int,
            default=8,
            help="num_workers to use on the CocoDetectionDataset (default %(default)s)",
        )
        parser.add_argument(
            "--no_augmentation", action="store_true", help="Do not use augmentation to train the model"
        )
        parser.add_argument(
            "--size",
            type=int,
            default=(None, None),
            nargs="+",
            help="If no augmentation (--no_augmentation) is used, --size can be used to resize all the frame.",
        )
        parser.add_argument(
            "--sample", action="store_true", help="Download a sample for train/val process (Default: %(default)s)"
        )
        return parent_parser

    def train_transform(self, frame, same_on_sequence: bool = True, same_on_frames: bool = False):
        if self.no_augmentation:
            if self.size[0] is not None and self.size[1] is not None:
                frame = T.Resize((self.size[0], self.size[1]))(frame)
            return frame.norm_resnet()

        scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
        frame = T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.RandomSelect(
                    T.RandomResizeWithAspectRatio(scales, max_size=1333),
                    T.Compose(
                        [
                            T.RandomResizeWithAspectRatio([400, 500, 600]),
                            T.RandomSizeCrop(384, 600),
                            T.RandomResizeWithAspectRatio(scales, max_size=1333),
                        ]
                    ),
                ),
            ],
            same_on_sequence=same_on_sequence,
            same_on_frames=same_on_frames,
        )(frame)

        return frame.norm_resnet()

    def val_transform(
        self, frame, same_on_sequence: bool = True, same_on_frames: bool = False,
    ):
        if self.no_augmentation:
            if self.size[0] is not None and self.size[1] is not None:
                frame = T.Resize((self.size[1], self.size[1]))(frame)
            return frame.norm_resnet()

        # Reszie keeping aspect ratio
        frame = T.RandomResizeWithAspectRatio(
            [800], max_size=1333, same_on_sequence=same_on_sequence, same_on_frames=same_on_frames
        )(frame)

        return frame.norm_resnet()

    def val_check(self):
        # Instance a default loader to set attributes
        self.coco_val = alodataset.CocoDetectionDataset(
            transform_fn=self.val_transform, sample=self.sample, **self.val_loader_kwargs,
        )
        self.sample = self.coco_val.sample or self.sample  # Update sample if user prompt is given
        self.CATEGORIES = self.coco_val.CATEGORIES if hasattr(self.coco_val, "CATEGORIES") else None
        self.labels_names = self.coco_val.labels_names if hasattr(self.coco_val, "labels_names") else None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            # Setup train/val loaders
            self.coco_train = alodataset.CocoDetectionDataset(
                transform_fn=self.train_transform, sample=self.sample, **self.train_loader_kwargs
            )
            self.coco_val = alodataset.CocoDetectionDataset(
                transform_fn=self.val_transform, sample=self.sample, **self.val_loader_kwargs
            )

    def train_dataloader(self):
        """Train dataloader"""
        # Init training loader
        if not hasattr(self, "coco_train"):
            self.setup()
        return self.coco_train.train_loader(batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self, sampler=None):
        """Val dataloader"""
        # Init training loader
        if not hasattr(self, "coco_val"):
            self.setup()
        return self.coco_val.train_loader(batch_size=self.batch_size, num_workers=self.num_workers, sampler=sampler)


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

    samples = next(iter(coco.val_dataloader()))
    samples[0].get_view().render()
