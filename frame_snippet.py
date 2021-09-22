from aloscene.bounding_boxes_2d import BoundingBoxes2D
import aloscene
import numpy as np


# frame = aloscene.Frame("/home/thibault/Desktop/yoga.jpg")
frame = aloscene.Frame(np.zeros((3, 256, 512)), normalization="01")
boxes3d = aloscene.BoundingBoxes3D([[10, 10, 10, 1.0, 1.0, 1.0, 0.0]])
frame.append_boxes3d(boxes3d)

frame.get_view().render()
