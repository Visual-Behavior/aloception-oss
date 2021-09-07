from numpy.core.fromnumeric import shape
import torch

from alodataset import BaseDataset, CocoDetectionDataset

from aloscene import Frame


class CocoSegmentationDataset(CocoDetectionDataset):
    def __init__(
        self,
        name: str = "coco",
        img_folder: str = None,
        ann_file: str = None,
        stuff_ann_file: str = None,
        things_classes: str = None,
        stuff_classes: str = None,
        **kwargs
    ):
        if "classes" in kwargs:
            raise Exception("Classes must be given in two parameters: 'things_classes' and 'stuff_classes'")

        super(CocoSegmentationDataset, self).__init__(
            name=name, img_folder=img_folder, ann_file=ann_file, classes=things_classes, **kwargs
        )
        if self.sample:
            return

        self.coco_stuff = CocoDetectionDataset(
            name=name, img_folder=img_folder, ann_file=stuff_ann_file, classes=stuff_classes, **kwargs
        )

    def getitem(self, idx):
        if self.sample:
            return BaseDataset.__getitem__(self, idx)

        things_boxes = super().getitem(idx)
        print(things_boxes, things_boxes.shape, things_boxes.names)
        stuff_boxes = self.coco_stuff.getitem(idx)
        print(stuff_boxes, stuff_boxes.shape, stuff_boxes.names)
        return torch.cat([things_boxes, stuff_boxes], dim=0)


if __name__ == "__main__":
    coco_dataset = CocoSegmentationDataset(
        img_folder="val2017",
        ann_file="annotations/instances_val2017.json",
        stuff_ann_file="annotations/stuff_val2017.json",
    )

    for f, frames in enumerate(coco_dataset.train_loader(batch_size=2)):
        frames = Frame.batch_list(frames)
        frames.get_view().render()
        if f > 1:
            break
