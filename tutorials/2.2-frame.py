from aloscene.renderer import View
from alodataset import WaymoDataset, Split
import torch

frame = WaymoDataset(
    split=Split.VAL,
    cameras=["front"],
    labels=["gt_boxes_2d"],
    sequence_size=2,  #  --> Let's use sequence of two frame
).get(42)["front"]

######
# The frame tensors and its labels
######

# Let's inspect the frame shape as well as the dim names.
# Named dim make it possible to track the frame dimension.
# It also come at cose since the number of possible operations with names on the frame get limited
# For instance, you con't reshape the tensor as you wish.
# print(frame.shape, frame.names)
# torch.Size([2, 3, 1280, 1920]) ('T', 'C', 'H', 'W')

# As we can se, the frame has one label: boxes2d
# This label is store as a dict with onw set of boxes: gt_boxes_2d
# This set of boxes is a list with two element (the size of the sequence)
# In the sequence, each set of boxes is maide of 11boxes.
print(frame)
# tensor(
# 	normalization=255,
# 	boxes2d={'gt_boxes_2d:[11, 11]'},

# If we Select the first frame of the sequence
# the list of boxes is now a simply a set of boxes instead of a list
frame_t0 = frame[0]
frame_t1 = frame[1]
print(frame_t0)
# tensor(
# 	normalization=255,
# 	boxes2d={gt_boxes_2d:torch.Size([11, 4])}
#   ....
#         [116., 115., 115.,  ...,  87.,  84.,  85.]]], names=('C', 'H', 'W'))

# tensor(
# 	normalization=255,
# 	boxes2d={gt_boxes_2d:torch.Size([11, 4])}


# You can't for now add one dimension on the frame using the pytorch api
# (the named dim will block the operation)
# > frame_t0 = torch.unsqueeze(frame_t0, dim=0) << Will throw an error
# Instead, you can call the `temporal method` to add a temporal dimension T on the frame.
n_frame_t0 = frame_t0.temporal()
n_frame_t1 = frame_t1.temporal()

# Now we can also concat the two frame together on the temporal dimension
frames = torch.cat([n_frame_t0, n_frame_t1], dim=0)

frames.get_view().render()
