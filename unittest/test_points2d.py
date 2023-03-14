# from aloscene.renderer import View
from alodataset import WaymoDataset  # , Split
import aloscene
import torch
import numpy as np


def test_crop_abs():
    image = np.zeros((3, 843, 1500))
    corners = [[298, 105], [1250, 105], [298, 705], [1250, 705]]
    frame = aloscene.Frame(image)
    labels = aloscene.Labels([0, 0, 0, 0], labels_names=["corners"])
    corners = aloscene.Points2D(
        corners, points_format="xy", frame_size=(frame.H, frame.W), absolute=True, labels=labels
    )
    frame.append_points2d(corners)
    frame = frame.crop(H_crop=(0.0, 0.5), W_crop=(0.0, 0.5))
    assert torch.allclose(frame.points2d[0].as_tensor(), corners[0].as_tensor())
    assert np.allclose(frame.points2d.frame_size[0], frame.HW[0])
    assert np.allclose(frame.points2d.frame_size[1], frame.HW[1])


if __name__ == "__main__":
    test_crop_abs()
