from aloscene.renderer import View
from alodataset import WaymoDataset, Split
import torch

waymo_dataset = WaymoDataset(split=Split.VAL, cameras=["front"], labels=["gt_boxes_2d"])

for data in waymo_dataset.stream_loader():

    front_t01 = data["front"]
    front_t02 = data["front"].hflip()

    front = torch.cat([front_t01, front_t02], dim=0)

    front.get_view(size=(500, 700)).render(View.CV)
