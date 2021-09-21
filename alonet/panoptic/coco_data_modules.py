"""`Pytorch Lightning Data Module <https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html>`_
connector between dataset and model. Load train/val/test sets and make the preprocessing required for use by the
architecture.
"""

from typing import Optional

import alodataset
import alonet


class CocoPanoptic2Detr(alonet.detr.CocoDetection2Detr):
    """Data connector between :mod:`~alodataset.CocoSegementationDataset` and :mod:`~alonet.panoptic.LitPanopticDetr`
    modules. See :mod:`~alonet.detr.CocoDetection2Detr` to see zall information about the methods.
    """

    def val_check(self):
        """Create a validation loader from sanity purposes.
        """
        # Instance a default loader to set attributes
        self.coco_val = alodataset.CocoSegementationDataset(
            transform_fn=self.val_transform, sample=self.sample, split=alodataset.Split.VAL,
        )
        self.sample = self.coco_val.sample or self.sample  # Update sample if user prompt is given
        self.label_names = self.coco_val.label_names if hasattr(self.coco_val, "label_names") else None

    def setup(self, stage: Optional[str] = None):
        """Called at the beginning of fit (train + validate), validate, test, and predict. This is a good hook when
        you need to build models dynamically or adjust something about them. This hook is called on every process when
        using DDP.

        Parameters
        ----------
        stage : Optional[str], optional
            Stage either `fit`, `validate`, `test` or `predict`, by default None
        """
        if stage == "fit" or stage is None:
            # Setup train/val loaders
            self.coco_train = alodataset.CocoSegementationDataset(
                transform_fn=self.val_transform if self.train_on_val else self.train_transform,
                sample=self.sample,
                split=alodataset.Split.VAL if self.train_on_val else alodataset.Split.TRAIN,
            )
            self.coco_val = alodataset.CocoSegementationDataset(
                transform_fn=self.val_transform, sample=self.sample, split=alodataset.Split.VAL,
            )


if __name__ == "__main__":
    # setup data
    coco = CocoPanoptic2Detr()
    coco.prepare_data()
    coco.setup()

    samples = next(iter(coco.val_dataloader()))
    samples[0].get_view().render()

    samples = next(iter(coco.train_dataloader()))
    samples[0].get_view().render()
