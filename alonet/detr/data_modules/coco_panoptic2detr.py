"""LightningDataModule that make the connection between :mod:`CocoPanopticDataset <alodataset.coco_panoptic_dataset>`
and :mod:`LitPanopticDetr <alonet.detr_panoptic.train>` modules. See
:mod:`Data2Detr <alonet.detr.data_modules.data2detr>` to more information about the methods and configurations.

Examples
--------
.. code-block:: python

    from alonet.detr import CocoPanoptic2Detr
    from aloscene import Frame

    datamodule = CocoPanoptic2Detr(sample = True)

    train_frame = next(iter(datamodule.train_dataloader()))
    train_frame = Frame.batch_list(train_frame).get_view().render()

    val_frame = next(iter(datamodule.val_dataloader()))
    val_frame = Frame.batch_list(val_frame).get_view().render()
"""

from typing import Optional

from alonet.detr.data_modules import Data2Detr
import alodataset


class CocoPanoptic2Detr(Data2Detr):
    def setup(self, stage: Optional[str] = None):
        """:attr:`train_dataset` and :attr:`val_dataset` datasets setup, follow the parameters used
        in class declaration. Also, set :attr:`label_names` attribute.

        Parameters
        ----------
        stage : str, optional
            Stage either `fit`, `validate`, `test` or `predict`, by default None
        """
        if stage == "fit" or stage is None:
            # Setup train/val loaders
            self.train_dataset = alodataset.CocoPanopticDataset(
                transform_fn=self.val_transform if self.train_on_val else self.train_transform,
                sample=self.sample,
                split=alodataset.Split.VAL if self.train_on_val else alodataset.Split.TRAIN,
            )
            self.sample = self.train_dataset.sample or self.sample  # Update sample if user prompt is given
            self.val_dataset = alodataset.CocoPanopticDataset(
                transform_fn=self.val_transform, sample=self.sample, split=alodataset.Split.VAL,
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
