from aloscene.renderer import View
from alodataset import ChairsSDHomDataset, Split
import torch

dataset = ChairsSDHomDataset(split=Split.VAL)

for frame in dataset.stream_loader():

    # Display everything
    frame.get_view(size=(500, 700)).render()

    # Display only flow
    frame.get_view([frame.flow], size=(500, 700)).render()

    # Display only flow, with some custom parameters
    # frame.get_view([frame.flow[0].__get_view__(convert_to_bgr=True, magnitude_max=50)], size=(500,700)).render()
    frame.get_view([frame.flow[0].get_view(convert_to_bgr=True, magnitude_max=50),]).render()

    # View only flow
    frame.flow[0].get_view([frame.flow[0]]).render()
