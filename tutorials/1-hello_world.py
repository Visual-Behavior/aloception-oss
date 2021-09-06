from aloscene.renderer import View
from alodataset import WaymoDataset, Split

waymo_dataset = WaymoDataset(split=Split.VAL, cameras=["front"], labels=["gt_boxes_2d"])

# Go through the loaded sequence
for data in waymo_dataset.stream_loader():
    # Select the front frame and display the result with opencv
    # `get_view()`, return a `View` object that can be be render using render()
    # 'cv' is pass to the render() method to use opencv instead of matplotlib (default)
    data["front"].get_view(size=(1280 // 2, 1920 // 2)).render(View.CV)
