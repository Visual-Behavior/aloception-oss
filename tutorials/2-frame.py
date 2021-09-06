from aloscene.renderer import View
from alodataset import WaymoDataset, Split

waymo_dataset = WaymoDataset(split=Split.VAL, cameras=["front"], labels=["gt_boxes_2d"])

# Get a frame at some position in the dataset
data = waymo_dataset.get(42)

# The frame object is a special type of Tensor (Labeled Tensor) with special attributes and labels
# attached to it
frame = data["front"]
# With the print method, one can see the attributes and labels attached to the tensor.
print(frame)
# tensor(
# 	normalization=255,
# 	boxes2d={'gt_boxes_2d:[11]'},
#    .....
#       names=('T', 'C', 'H', 'W'))


# Render the frame
data["front"].get_view().render()
