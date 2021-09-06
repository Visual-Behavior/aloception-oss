from aloscene.renderer import View
from alodataset import WaymoDataset, Split

waymo_dataset = WaymoDataset(split=Split.VAL, cameras=["front"], labels=["gt_boxes_2d"])

for data in waymo_dataset.stream_loader():
    data["front"].get_view(size=(500, 700)).render(View.CV)
