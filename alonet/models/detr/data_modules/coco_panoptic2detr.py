from typing import Optional

from alonet.detr.data_modules import Data2Detr
import alodataset


class CocoPanoptic2Detr(Data2Detr):
    def setup(self, stage: Optional[str] = None, fix_classes_len: int = 250):
        """:attr:`train_dataset` and :attr:`val_dataset` datasets setup, follow the parameters used
        in class declaration. Also, set :attr:`label_names` attribute.

        Parameters
        ----------
        stage : str, optional
            Stage either `fit`, `validate`, `test` or `predict`, by default None
        fix_classes_len : int, optional
            Fix datasets to a specific number the number of classes, filling the rest with "N/A" value.
        """
        if stage == "fit" or stage is None:
            # Setup train/val loaders
            self.train_dataset = alodataset.CocoPanopticDataset(
                transform_fn=self.val_transform if self.train_on_val else self.train_transform,
                sample=self.sample,
                split=alodataset.Split.VAL if self.train_on_val else alodataset.Split.TRAIN,
                fix_classes_len=fix_classes_len,
            )
            self.sample = self.train_dataset.sample or self.sample  # Update sample if user prompt is given
            self.val_dataset = alodataset.CocoPanopticDataset(
                transform_fn=self.val_transform,
                sample=self.sample,
                split=alodataset.Split.VAL,
                fix_classes_len=fix_classes_len,
            )
            self.sample = self.val_dataset.sample or self.sample  # Update sample if user prompt is given
            self.label_names = self.val_dataset.label_names if hasattr(self.val_dataset, "label_names") else None


if __name__ == "__main__":
    # setup data
    coco = CocoPanoptic2Detr()
    coco.prepare_data()
    coco.setup()

    samples = next(iter(coco.train_dataloader()))
    samples[0].get_view().render()

    samples = next(iter(coco.val_dataloader()))
    samples[0].get_view().render()
