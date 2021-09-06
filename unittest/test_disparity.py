import numpy as np
import torch
import aloscene

from aloscene import Frame, Disparity


def test_hflip():

    S = 600
    frame = aloscene.Frame(np.random.uniform(0, 1, (3, 600, 600)), names=("C", "H", "W"))

    disp_mask1 = aloscene.Mask(torch.zeros((1, frame.H, frame.W)), names=("C", "H", "W"))
    disp_mask1[:, :, :300] = 1
    disp_mask2 = aloscene.Mask(torch.zeros((1, frame.H, frame.W)), names=("C", "H", "W"))
    disp_mask2[:, :, 300:] = 1

    disp = aloscene.Disparity(torch.zeros((1, frame.H, frame.W)), names=("C", "H", "W"))
    disp.append_occlusion(disp_mask1, "occ1")
    disp.append_occlusion(disp_mask2, "occ2")

    frame.append_disparity(disp)
    frame.hflip()


def test_signed_unsigned():
    H = 3
    W = 4
    data = np.arange(H * W).reshape((1, 1, H, W))
    disp_left = Disparity(data, names=("T", "C", "H", "W"), camera_side="left", disp_format="unsigned")
    disp_right = Disparity(data.copy(), names=("T", "C", "H", "W"), camera_side="right", disp_format="unsigned")

    # test signed
    signed_left = disp_left.signed()
    assert np.array_equal(-1 * data, signed_left.numpy())
    assert signed_left.disp_format == "signed"

    signed_right = disp_right.signed()
    assert np.array_equal(data, signed_right.numpy())
    assert signed_right.disp_format == "signed"

    # test unsigned
    unsigned_left = disp_left.unsigned()
    assert np.array_equal(data, unsigned_left.numpy())
    assert unsigned_left.disp_format == "unsigned"

    unsigned_right = disp_right.unsigned()
    assert np.array_equal(data, unsigned_right.numpy())
    assert unsigned_right.disp_format == "unsigned"

    # test signed followed by unsigned
    signed_unsigned_left = disp_left.signed().unsigned()
    assert np.array_equal(data, signed_unsigned_left.numpy())
    assert signed_unsigned_left.disp_format == "unsigned"

    signed_unsigned_right = disp_right.signed().unsigned()
    assert np.array_equal(data, signed_unsigned_right.numpy())
    assert signed_unsigned_right.disp_format == "unsigned"

    # test unsigned folled by signed
    unsigned_signed_left = disp_left.unsigned().signed()
    assert np.array_equal(-1 * data, unsigned_signed_left.numpy())
    assert unsigned_signed_left.disp_format == "signed"

    unsigned_signed_right = disp_right.unsigned().signed()
    assert np.array_equal(data, unsigned_signed_right.numpy())
    assert unsigned_signed_right.disp_format == "signed"


if __name__ == "__main__":
    test_signed_unsigned()
    test_hflip()
