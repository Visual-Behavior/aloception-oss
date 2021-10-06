"""`Pytorch Lightning Data Module <https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html>`_
connector between dataset and model. Load train/val/test sets and make the preprocessing required for use by the
architecture. Make the connection between :mod:`~alodataset.CocoPanopticDataset` and
:mod:`~alonet.panoptic.LitPanopticDetr` modules. See :mod:`~alonet.detr.Data2Detr` to see all information
about the methods.
"""

from typing import Optional

from alonet.detr.data_modules import Data2Detr
import alodataset


class CocoPanoptic2Detr(Data2Detr):
    def setup(self, stage: Optional[str] = None, fix_classes_len: int = 250):
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
