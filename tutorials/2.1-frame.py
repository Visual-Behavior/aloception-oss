from aloscene.renderer import View
from alodataset import WaymoDataset, Split

frame = WaymoDataset(split=Split.VAL, cameras=["front"], labels=["gt_boxes_2d"]).get(42)["front"]

######
# Here are some example of the operations that can be done on a frame
######

# Normalize the values between 0 and 1
frame_01 = frame.norm01()
print(frame_01)
# tensor(
# 	normalization=01,
# 	boxes2d={'gt_boxes_2d:[11]'},


# Normalize the values between 0 and 255
frame_255 = frame.norm255()
print(frame_255)
# tensor(
# 	normalization=255,
# 	boxes2d={'gt_boxes_2d:[11]'},


# Get the frame ready to be send into a resnet backbone
frame_resnet = frame.norm_resnet()
print(frame_resnet)
# tensor(
# 	normalization=resnet, mean_std=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
# 	boxes2d={'gt_boxes_2d:[11]'},


# Custom mean/std normalization. Note: the mean/std must be normalized
# with values within the range 0 and 1
my_frame = frame.mean_std_norm((0.42, 0.4, 0.40), (0.41, 0.2, 0.45), name="my_norm")
print(my_frame)
# tensor(
# 	normalization=my_norm, mean_std=((0.42, 0.4, 0.4), (0.41, 0.2, 0.45)),
# 	boxes2d={'gt_boxes_2d:[11]'},


# Note that even after all the normalization, you can still render your frame properly
# at any moment
my_frame.get_view().render()
