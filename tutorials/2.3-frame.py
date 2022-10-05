from aloscene.renderer import View
from alodataset import WaymoDataset, Split
import torch

frame = WaymoDataset(split=Split.VAL, cameras=["front"], labels=["gt_boxes_2d"], sequence_size=1,).get(
    42
)["front"]

######
# Special & usefull operations on the frame
######

# We can resize the frame using the following method.
# This will automaticly resize the boxes accordingly if needed
# Note that is you call F.resize instead of frame.resize() this will only change
# the content of the tensor without changing the labels
# TODO: Raise a warning is this is the case ?
resized_frame = frame.resize((100, 200))
resized_frame.get_view().render()

# Crop a frame
# When croping, the associated label will be cropped to if needed
cropped_frame = frame[0, :, 400:900, 500:800]
print("cropped_frame", cropped_frame.shape, cropped_frame.names)
# > cropped_frame torch.Size([3, 500, 300]) ('C', 'H', 'W')
# Additionally, you can also use F.crop on the frame tensor.


# Horizontal flip
cropped_frame_hflip = cropped_frame.hflip()

# Render both frame
cropped_frame.get_view().add(cropped_frame_hflip.get_view()).render()
