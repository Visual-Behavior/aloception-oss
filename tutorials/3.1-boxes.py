from aloscene.renderer import View
from alodataset import WaymoDataset, Split
import torch

frames = WaymoDataset(split=Split.VAL, cameras=["front"], labels=["gt_boxes_2d"], sequence_size=2,).get(
    42
)["front"]

######
# Boxes manipulation
######

boxes_t0 = frames[0].boxes2d["gt_boxes_2d"]
boxes_t1 = frames[1].boxes2d["gt_boxes_2d"]

print(boxes_t0.shape, boxes_t0)
# torch.Size([11, 4]) tensor(
# 	boxes_format=xcyc, absolute=True, frame_size=(1280, 1920),
# 	labels=torch.Size([11, 1])
# ...

boxes = torch.cat([boxes_t0, boxes_t1], dim=0)
print(boxes.shape, boxes)
# torch.Size([22, 4]) tensor(
# 	boxes_format=xcyc, absolute=True, frame_size=(1280, 1920),
# 	labels=torch.Size([22, 1])
# ...
