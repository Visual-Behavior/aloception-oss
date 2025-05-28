import alodataset
from alonet.models.detr.data_modules.data2detr import Data2Detr


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
        name: str = "coco",
        classes: list = None,
        train_folder: str = "train2017",
        train_ann: str = "annotations/instances_train2017.json",
        val_folder: str = "val2017",
        val_ann: str = "annotations/instances_val2017.json",
        return_masks: bool = False,
        train_on_val: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
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
        self.train_on_val = train_on_val

        if self.train_on_val:
            self.train_loader_kwargs["img_folder"] = val_folder
            self.train_loader_kwargs["ann_file"] = val_ann

    def setup_train_dataset(self) -> alodataset.CocoBaseDataset:
        return alodataset.CocoBaseDataset(
            transform_fn=self.train_transform, sample=self.sample, **self.train_loader_kwargs
        )

    def setup_val_dataset(self) -> alodataset.CocoBaseDataset:
        return alodataset.CocoBaseDataset(
            transform_fn=self.val_transform, sample=self.sample, **self.val_loader_kwargs
        )


if __name__ == "__main__":
    # setup data
    loader_kwargs = dict(
        batch_size=1,
        num_workers=1,
        name="coco",
        train_folder="train2017",
        train_ann="annotations/instances_train2017.json",
        val_folder="val2017",
        val_ann="annotations/instances_val2017.json",
    )

    coco = CocoDetection2Detr(**loader_kwargs)
    coco.setup(stage="training")
    iterator = iter(coco.train_dataloader)
    for i in range(2):
        samples = next(iterator)
        samples[0].get_view().render()
