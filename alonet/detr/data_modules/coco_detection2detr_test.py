from argparse import ArgumentParser, Namespace
from typing import Optional

from alonet.detr.data_modules import Data2Detr
import alodataset
from alonet.common.base_datamodule import BaseDataModule
import aloscene
import alodataset.transforms as T


class CocoDetection2Detr(BaseDataModule):
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
            self.train_dataset = alodataset.CocoBaseDataset(
                transform_fn=self.train_transform, sample=self.sample, **self.train_loader_kwargs
            )
            # Update sample if user prompt is given
            self.sample = self.train_dataset.sample or self.sample
            self.val_dataset = alodataset.CocoBaseDataset(
                transform_fn=self.val_transform, sample=self.sample, **self.val_loader_kwargs
            )
            # Update sample if user prompt is given
            self.sample = self.val_dataset.sample or self.sample
            self.label_names = self.val_dataset.label_names if hasattr(
                self.val_dataset, "label_names") else None

    def _train_transform_no_aug(self, frames):
        if self.size[0] is not None and self.size[1] is not None:
            frame = T.Resize((self.size[0], self.size[1]))(frame)
            return frame.norm_resnet()

    def _train_transform_aug(self, frame: aloscene.Frame, same_on_sequence: bool = True, same_on_frames: bool = False):
        """Transorm requered to train on each frame

        Parameters
        ----------
        frame : :mod:`~aloscene.frame`
            Input frame to transform
        same_on_sequence : bool, optional
            Use same data augmentation size of each sequence, by default True
        same_on_frames : bool, optional
            Use same data augmentation size of each frame, by default False

        Returns
        -------
        :mod:`~aloscene.frame`
            Frame with respective changes by transform function
        """

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
        self, frame: aloscene.Frame, same_on_sequence: bool = True, same_on_frames: bool = False,
    ):
        """Transform requered to valid on each frame

        Parameters
        ----------
        frame : :mod:`~aloscene.frame`
            Input frame to transform
        same_on_sequence : bool, optional
            Use same data augmentation size of each sequence, by default True
        same_on_frames : bool, optional
            Use same data augmentation size of each frame, by default False

        Returns
        -------
        :mod:`~aloscene.frame`
            Frame with respective changes by transform function
        """
        # Reszie keeping aspect ratio
        frame = T.RandomResizeWithAspectRatio(
            [800], max_size=1333, same_on_sequence=same_on_sequence, same_on_frames=same_on_frames
        )(frame)

        return frame.norm_resnet()


if __name__ == "__main__":
    # setup data
    loader_kwargs = dict(
        name="coco",
        train_folder="train2017",
        train_ann="annotations/instances_train2017.json",
        val_folder="val2017",
        val_ann="annotations/instances_val2017.json",
    )

    args = CocoDetection2Detr.add_argparse_args(
        ArgumentParser()).parse_args()  # Help provider
    coco = CocoDetection2Detr(args, **loader_kwargs)
    coco.prepare_data()
    coco.setup()
    iterator = iter(coco.train_dataloader())
    for i in range(2):
        samples = next(iterator)
        samples[0].get_view().render()
