from aloscene.renderer import View
from alodataset import WaymoDataset, Split
import torch

waymo_dataset = WaymoDataset(split=Split.VAL, cameras=["front"], labels=["gt_boxes_2d"])

for data in waymo_dataset.stream_loader():

    front_t01 = data["front"]
    front_t02 = data["front"].hflip()

    front = torch.cat([front_t01, front_t02], dim=0)

    # Display only boxes2D
    front.get_view([front.boxes2d], size=(500, 700)).render()

    # Display everything but boxes2D
    front.get_view(exclude=[front.boxes2d], size=(500, 700)).render()

    # Display everything only boxes with custom params
    front.get_view([front.boxes2d["gt_boxes_2d"][0].get_view(front[0])], size=(500, 700)).render()

    # View composition
    # Display only boxes2D
    front.get_view([front.boxes2d], size=(500, 700)).add(front.get_view(exclude=[front.boxes2d], size=(500, 700))).add(
        front.get_view([front.boxes2d["gt_boxes_2d"][0].get_view(front[0])], size=(500, 700))
    ).render()
