from cv2 import detail_SphericalProjector
from aloscene import Depth
import torch

dpth = Depth(torch.ones((2, 5, 5)), is_absolute=False)
mask = torch.ones((2, 5, 5))

# depth = depth.encode_abspolute()
# print(dpth[mask].shape)


dpth = dpth.encode_absolute()