from alonet.models.detr.data_modules.data2detr import Data2Detr
import alodataset


class CocoPanoptic2Detr(Data2Detr):
    def setup_train_dataset(self) -> alodataset.CocoPanopticDataset:
        return alodataset.CocoPanopticDataset(
            name="coco",
            split=alodataset.Split.TRAIN,
            return_masks=True,
            transform_fn=self.train_transform,
        )

    def setup_val_dataset(self) -> alodataset.CocoPanopticDataset:
        return alodataset.CocoPanopticDataset(
            name="coco",
            split=alodataset.Split.VAL,
            return_masks=True,
            transform_fn=self.val_transform,
        )


if __name__ == "__main__":
    # setup data
    coco = CocoPanoptic2Detr(batch_size=1, num_workers=1)
    coco.setup(stage="training")

    samples = next(iter(coco.train_dataloader))
    samples[0].get_view().render()

    samples = next(iter(coco.val_dataloader))
    samples[0].get_view().render()
