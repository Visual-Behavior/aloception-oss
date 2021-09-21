from typing import Optional

import alodataset
import alonet


class CocoPanoptic2Detr(alonet.detr.CocoDetection2Detr):
    def val_check(self):
        # Instance a default loader to set attributes
        self.coco_val = alodataset.CocoSegementationDataset(
            transform_fn=self.val_transform, sample=self.sample, split=alodataset.Split.VAL,
        )
        self.sample = self.coco_val.sample or self.sample  # Update sample if user prompt is given
        self.label_names = self.coco_val.label_names if hasattr(self.coco_val, "label_names") else None

    def setup(self, stage: Optional[str] = None) -> None:
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
