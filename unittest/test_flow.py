import numpy as np
import torch

from aloscene.frame import Frame
from aloscene.flow import Flow


def test_flow_hflip():
    # create a frame with a 3 x 3 image
    image = torch.full((1, 3, 3, 3), 255, dtype=torch.float32)
    frame = Frame(image, normalization="255", names=("T", "C", "H", "W"))

    # create Ground-Truth original flow
    flow_np = np.arange(1, 10).reshape(3, 3)
    flow_np = np.repeat(flow_np[np.newaxis, np.newaxis, ...], 2, axis=1)
    flow_torch = torch.from_numpy(flow_np.astype(np.float32))
    flow = Flow(flow_torch, names=("T", "C", "H", "W"))
    frame.append_flow(flow)

    # create Ground-Truth horizontally flipped flow
    flow_flip_GT = np.empty_like(flow_np)
    flow_flip_GT[0, 0, ...] = [[-3, -2, -1], [-6, -5, -4], [-9, -8, -7]]
    flow_flip_GT[0, 1, ...] = [[3, 2, 1], [6, 5, 4], [9, 8, 7]]

    # compute flow flipped
    frame_flip = frame.hflip()
    flow_flip = frame_flip.flow.numpy()

    # compare ground truth to result
    assert np.allclose(flow_flip_GT, flow_flip)


if __name__ == "__main__":
    test_flow_hflip()
