from aloscene.renderer import View
from alodataset import WaymoDataset, Split
import torch

frames = WaymoDataset(split=Split.VAL, cameras=["front"], labels=["gt_boxes_2d"], sequence_size=2,).get(  #
    2
)["front"]

######
# Boxes
######

# Each frame can have multiple set of 2D boxes ("gt", "pred", ...)
# The following line (both do the same) get the set of boxes from the first frame of the sequence
boxes = frames[0].boxes2d["gt_boxes_2d"]
# The same can be done without slicing the frames but by slicing the boxes (to select the first set of boxes in the sequence)
boxes = frames.boxes2d["gt_boxes_2d"][0]

print("boxes", boxes)
# boxes tensor(
# 	boxes_format=xcyc, absolute=True, labels={}, frame_size=(1280, 1920),
# ....

# To view the boxes, a frame must be passed to the get_view method
boxes.get_view(frames[0]).render()
