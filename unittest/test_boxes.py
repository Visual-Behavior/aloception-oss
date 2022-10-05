# from aloscene.renderer import View
from alodataset import WaymoDataset  # , Split
import aloscene
import torch
import numpy as np

waymo_dataset = WaymoDataset(sample=True)
TEST_FRAME = 0


def boxes_equal(boxes1, boxes2):
    if torch.mean((boxes1 - boxes2).rename_(None).abs()) < 1e-4:
        return True
    else:
        return False


def test_boxes_from_dt():
    boxes = waymo_dataset.get(TEST_FRAME)["front"].boxes2d["gt_boxes_2d"]
    assert isinstance(boxes, list)
    assert len(boxes) == 2


def test_boxes_rel_xcyc():
    boxes = waymo_dataset.get(TEST_FRAME)["front"].boxes2d["gt_boxes_2d"][0]

    boxes = boxes.rel_pos().xcyc()

    boxes_rel_yxyx = boxes.rel_pos().yxyx()
    assert boxes_rel_yxyx.absolute == False
    assert boxes_rel_yxyx.boxes_format == "yxyx"

    boxes_rel_xyxy = boxes.rel_pos().xyxy()
    assert boxes_rel_xyxy.absolute == False
    assert boxes_rel_xyxy.boxes_format == "xyxy"

    boxes_rel_xcyc = boxes.rel_pos().xcyc()
    assert boxes_rel_xcyc.absolute == False
    assert boxes_rel_xcyc.boxes_format == "xcyc"

    boxes_abs_yxyx = boxes.abs_pos((100, 200)).yxyx()
    assert boxes_abs_yxyx.frame_size == (100, 200)
    assert boxes_abs_yxyx.absolute == True
    assert boxes_abs_yxyx.boxes_format == "yxyx"

    boxes_abs_xyxy = boxes.abs_pos((100, 200)).xyxy()
    assert boxes_abs_xyxy.frame_size == (100, 200)
    assert boxes_abs_xyxy.absolute == True
    assert boxes_abs_xyxy.boxes_format == "xyxy"

    boxes_abs_xcyc = boxes.abs_pos((100, 200)).xcyc()
    assert boxes_abs_xcyc.frame_size == (100, 200)
    assert boxes_abs_xcyc.absolute == True
    assert boxes_abs_xcyc.boxes_format == "xcyc"

    # Check back equality
    assert boxes_equal(boxes_rel_yxyx.rel_pos().xcyc(), boxes)
    assert boxes_equal(boxes_rel_xyxy.rel_pos().xcyc(), boxes)
    assert boxes_equal(boxes_rel_xcyc.rel_pos().xcyc(), boxes)
    assert boxes_equal(boxes_abs_xyxy.rel_pos().xcyc(), boxes)
    assert boxes_equal(boxes_abs_yxyx.rel_pos().xcyc(), boxes)
    assert boxes_equal(boxes_abs_xcyc.rel_pos().xcyc(), boxes)


def test_boxes_rel_yxyx():
    boxes = waymo_dataset.get(TEST_FRAME)["front"].boxes2d["gt_boxes_2d"][0]

    boxes = boxes.rel_pos().yxyx()

    boxes_rel_yxyx = boxes.rel_pos().yxyx()
    assert boxes_rel_yxyx.absolute == False
    assert boxes_rel_yxyx.boxes_format == "yxyx"

    boxes_rel_xyxy = boxes.rel_pos().xyxy()
    assert boxes_rel_xyxy.absolute == False
    assert boxes_rel_xyxy.boxes_format == "xyxy"

    boxes_rel_xcyc = boxes.rel_pos().xcyc()
    assert boxes_rel_xcyc.absolute == False
    assert boxes_rel_xcyc.boxes_format == "xcyc"

    boxes_abs_yxyx = boxes.abs_pos((100, 200)).yxyx()
    assert boxes_abs_yxyx.frame_size == (100, 200)
    assert boxes_abs_yxyx.absolute == True
    assert boxes_abs_yxyx.boxes_format == "yxyx"

    boxes_abs_xyxy = boxes.abs_pos((100, 200)).xyxy()
    assert boxes_abs_xyxy.frame_size == (100, 200)
    assert boxes_abs_xyxy.absolute == True
    assert boxes_abs_xyxy.boxes_format == "xyxy"

    boxes_abs_xcyc = boxes.abs_pos((100, 200)).xcyc()
    assert boxes_abs_xcyc.frame_size == (100, 200)
    assert boxes_abs_xcyc.absolute == True
    assert boxes_abs_xcyc.boxes_format == "xcyc"

    # Check back equality
    assert boxes_equal(boxes_rel_yxyx.rel_pos().yxyx(), boxes)
    assert boxes_equal(boxes_rel_xyxy.rel_pos().yxyx(), boxes)
    assert boxes_equal(boxes_rel_xcyc.rel_pos().yxyx(), boxes)
    assert boxes_equal(boxes_abs_xyxy.rel_pos().yxyx(), boxes)
    assert boxes_equal(boxes_abs_yxyx.rel_pos().yxyx(), boxes)
    assert boxes_equal(boxes_abs_xcyc.rel_pos().yxyx(), boxes)


def test_boxes_rel_xyxy():
    boxes = waymo_dataset.get(TEST_FRAME)["front"].boxes2d["gt_boxes_2d"][0]

    boxes = boxes.rel_pos().xyxy()

    boxes_rel_yxyx = boxes.rel_pos().yxyx()
    assert boxes_rel_yxyx.absolute == False
    assert boxes_rel_yxyx.boxes_format == "yxyx"

    boxes_rel_xyxy = boxes.rel_pos().xyxy()
    assert boxes_rel_xyxy.absolute == False
    assert boxes_rel_xyxy.boxes_format == "xyxy"

    boxes_rel_xcyc = boxes.rel_pos().xcyc()
    assert boxes_rel_xcyc.absolute == False
    assert boxes_rel_xcyc.boxes_format == "xcyc"

    boxes_abs_yxyx = boxes.abs_pos((100, 200)).yxyx()
    assert boxes_abs_yxyx.frame_size == (100, 200)
    assert boxes_abs_yxyx.absolute == True
    assert boxes_abs_yxyx.boxes_format == "yxyx"

    boxes_abs_xyxy = boxes.abs_pos((100, 200)).xyxy()
    assert boxes_abs_xyxy.frame_size == (100, 200)
    assert boxes_abs_xyxy.absolute == True
    assert boxes_abs_xyxy.boxes_format == "xyxy"

    boxes_abs_xcyc = boxes.abs_pos((100, 200)).xcyc()
    assert boxes_abs_xcyc.frame_size == (100, 200)
    assert boxes_abs_xcyc.absolute == True
    assert boxes_abs_xcyc.boxes_format == "xcyc"

    # Check back equality
    assert boxes_equal(boxes_rel_yxyx.rel_pos().xyxy(), boxes)
    assert boxes_equal(boxes_rel_xyxy.rel_pos().xyxy(), boxes)
    assert boxes_equal(boxes_rel_xcyc.rel_pos().xyxy(), boxes)
    assert boxes_equal(boxes_abs_xyxy.rel_pos().xyxy(), boxes)
    assert boxes_equal(boxes_abs_yxyx.rel_pos().xyxy(), boxes)
    assert boxes_equal(boxes_abs_xcyc.rel_pos().xyxy(), boxes)


def test_boxes_abs_xcyc():
    boxes = waymo_dataset.get(TEST_FRAME)["front"].boxes2d["gt_boxes_2d"][0]

    boxes = boxes.abs_pos((400, 500)).xcyc()

    boxes_rel_yxyx = boxes.rel_pos().yxyx()
    assert boxes_rel_yxyx.absolute == False
    assert boxes_rel_yxyx.boxes_format == "yxyx"

    boxes_rel_xyxy = boxes.rel_pos().xyxy()
    assert boxes_rel_xyxy.absolute == False
    assert boxes_rel_xyxy.boxes_format == "xyxy"

    boxes_rel_xcyc = boxes.rel_pos().xcyc()
    assert boxes_rel_xcyc.absolute == False
    assert boxes_rel_xcyc.boxes_format == "xcyc"

    boxes_abs_yxyx = boxes.abs_pos((100, 200)).yxyx()
    assert boxes_abs_yxyx.frame_size == (100, 200)
    assert boxes_abs_yxyx.absolute == True
    assert boxes_abs_yxyx.boxes_format == "yxyx"

    boxes_abs_xyxy = boxes.abs_pos((100, 200)).xyxy()
    assert boxes_abs_xyxy.frame_size == (100, 200)
    assert boxes_abs_xyxy.absolute == True
    assert boxes_abs_xyxy.boxes_format == "xyxy"

    boxes_abs_xcyc = boxes.abs_pos((100, 200)).xcyc()
    assert boxes_abs_xcyc.frame_size == (100, 200)
    assert boxes_abs_xcyc.absolute == True
    assert boxes_abs_xcyc.boxes_format == "xcyc"

    assert boxes_equal(boxes_rel_yxyx.abs_pos((400, 500)).xcyc(), boxes)
    assert boxes_equal(boxes_rel_xyxy.abs_pos((400, 500)).xcyc(), boxes)
    assert boxes_equal(boxes_rel_xcyc.abs_pos((400, 500)).xcyc(), boxes)
    assert boxes_equal(boxes_abs_xyxy.abs_pos((400, 500)).xcyc(), boxes)
    assert boxes_equal(boxes_abs_yxyx.abs_pos((400, 500)).xcyc(), boxes)
    assert boxes_equal(boxes_abs_xcyc.abs_pos((400, 500)).xcyc(), boxes)


def test_boxes_abs_yxyx():
    boxes = waymo_dataset.get(TEST_FRAME)["front"].boxes2d["gt_boxes_2d"][0]

    boxes = boxes.abs_pos((400, 500)).yxyx()

    boxes_rel_yxyx = boxes.rel_pos().yxyx()
    assert boxes_rel_yxyx.absolute == False
    assert boxes_rel_yxyx.boxes_format == "yxyx"

    boxes_rel_xyxy = boxes.rel_pos().xyxy()
    assert boxes_rel_xyxy.absolute == False
    assert boxes_rel_xyxy.boxes_format == "xyxy"

    boxes_rel_xcyc = boxes.rel_pos().xcyc()
    assert boxes_rel_xcyc.absolute == False
    assert boxes_rel_xcyc.boxes_format == "xcyc"

    boxes_abs_yxyx = boxes.abs_pos((100, 200)).yxyx()
    assert boxes_abs_yxyx.frame_size == (100, 200)
    assert boxes_abs_yxyx.absolute == True
    assert boxes_abs_yxyx.boxes_format == "yxyx"

    boxes_abs_xyxy = boxes.abs_pos((100, 200)).xyxy()
    assert boxes_abs_xyxy.frame_size == (100, 200)
    assert boxes_abs_xyxy.absolute == True
    assert boxes_abs_xyxy.boxes_format == "xyxy"

    boxes_abs_xcyc = boxes.abs_pos((100, 200)).xcyc()
    assert boxes_abs_xcyc.frame_size == (100, 200)
    assert boxes_abs_xcyc.absolute == True
    assert boxes_abs_xcyc.boxes_format == "xcyc"

    # Check back equality
    assert boxes_equal(boxes_rel_yxyx.abs_pos((400, 500)).yxyx(), boxes)
    assert boxes_equal(boxes_rel_xyxy.abs_pos((400, 500)).yxyx(), boxes)
    assert boxes_equal(boxes_rel_xcyc.abs_pos((400, 500)).yxyx(), boxes)
    assert boxes_equal(boxes_abs_xyxy.abs_pos((400, 500)).yxyx(), boxes)
    assert boxes_equal(boxes_abs_yxyx.abs_pos((400, 500)).yxyx(), boxes)
    assert boxes_equal(boxes_abs_xcyc.abs_pos((400, 500)).yxyx(), boxes)


def test_boxes_abs_xyxy():
    boxes = waymo_dataset.get(TEST_FRAME)["front"].boxes2d["gt_boxes_2d"][0]

    boxes = boxes.abs_pos((400, 500)).xyxy()

    boxes_rel_yxyx = boxes.rel_pos().yxyx()
    assert boxes_rel_yxyx.absolute == False
    assert boxes_rel_yxyx.boxes_format == "yxyx"

    boxes_rel_xyxy = boxes.rel_pos().xyxy()
    assert boxes_rel_xyxy.absolute == False
    assert boxes_rel_xyxy.boxes_format == "xyxy"

    boxes_rel_xcyc = boxes.rel_pos().xcyc()
    assert boxes_rel_xcyc.absolute == False
    assert boxes_rel_xcyc.boxes_format == "xcyc"

    boxes_abs_yxyx = boxes.abs_pos((100, 200)).yxyx()
    assert boxes_abs_yxyx.frame_size == (100, 200)
    assert boxes_abs_yxyx.absolute == True
    assert boxes_abs_yxyx.boxes_format == "yxyx"

    boxes_abs_xyxy = boxes.abs_pos((100, 200)).xyxy()
    assert boxes_abs_xyxy.frame_size == (100, 200)
    assert boxes_abs_xyxy.absolute == True
    assert boxes_abs_xyxy.boxes_format == "xyxy"

    boxes_abs_xcyc = boxes.abs_pos((100, 200)).xcyc()
    assert boxes_abs_xcyc.frame_size == (100, 200)
    assert boxes_abs_xcyc.absolute == True
    assert boxes_abs_xcyc.boxes_format == "xcyc"

    # Check back equality
    assert boxes_equal(boxes_rel_yxyx.abs_pos((400, 500)).xyxy(), boxes)
    assert boxes_equal(boxes_rel_xyxy.abs_pos((400, 500)).xyxy(), boxes)
    assert boxes_equal(boxes_rel_xcyc.abs_pos((400, 500)).xyxy(), boxes)
    assert boxes_equal(boxes_abs_xyxy.abs_pos((400, 500)).xyxy(), boxes)
    assert boxes_equal(boxes_abs_yxyx.abs_pos((400, 500)).xyxy(), boxes)
    assert boxes_equal(boxes_abs_xcyc.abs_pos((400, 500)).xyxy(), boxes)


def _test_padded_boxes():
    """Outdated"""

    boxes_abs = aloscene.BoundingBoxes2D(
        [[20, 40, 40, 60], [40, 60, 60, 80]], boxes_format="xyxy", absolute=True, frame_size=(80, 80)
    )

    boxes_rel = boxes_abs.rel_pos()

    assert boxes_rel[0][0] == 0.25
    assert boxes_rel[0][1] == 0.5
    assert boxes_rel[0][2] == 0.5
    assert boxes_rel[0][3] == 0.75
    assert boxes_rel[1][0] == 0.5
    assert boxes_rel[1][1] == 0.75
    assert boxes_rel[1][2] == 0.75
    assert boxes_rel[1][3] == 1.0

    boxes_abs_padded = boxes_abs.pad(offset_y=(0, 1.0), offset_x=(0, 3.0), pad_boxes=True)
    boxes_rel_padded = boxes_rel.pad(offset_y=(0, 1.0), offset_x=(0, 3.0), pad_boxes=True)

    assert boxes_equal(boxes_abs_padded, boxes_abs)
    assert not boxes_equal(boxes_rel_padded, boxes_abs)

    assert boxes_abs_padded.frame_size == (80 * 2, 80 * 4)
    assert boxes_abs_padded.padded_size == None
    assert boxes_rel_padded.frame_size == None
    assert boxes_rel_padded.padded_size == None

    boxes_abs_padded = boxes_abs.pad(offset_y=(0, 1.0), offset_x=(0, 3.0), pad_boxes=False)
    boxes_rel_padded = boxes_rel.pad(offset_y=(0, 1.0), offset_x=(0, 3.0), pad_boxes=False)

    assert boxes_abs_padded.frame_size == (80, 80)

    assert boxes_abs_padded.padded_size == ((0, 1.0), (0, 3.0))
    assert boxes_rel_padded.frame_size == None
    assert boxes_rel_padded.padded_size == ((0, 1.0), (0, 3.0))

    boxes_abs_padded = boxes_abs_padded.pad(offset_y=(0, 1.0), offset_x=(0, 1.0), pad_boxes=False)
    boxes_rel_padded = boxes_rel_padded.pad(offset_y=(0, 1.0), offset_x=(0, 1.0), pad_boxes=False)

    assert boxes_abs_padded.frame_size == (80, 80)
    assert boxes_abs_padded.padded_size == ((0, 3.0), (0, 7.0))
    assert boxes_rel_padded.frame_size == None
    assert boxes_rel_padded.padded_size == (400, 800)

    n_boxes_abs_padded = boxes_rel_padded.abs_pos((200, 200))
    assert n_boxes_abs_padded.frame_size == (200, 200)
    assert n_boxes_abs_padded.padded_size == (200 * 2 * 2, 200 * 4 * 2)
    n_boxes_abs_padded = n_boxes_abs_padded.abs_pos((100, 100))
    assert n_boxes_abs_padded.frame_size == (100, 100)
    assert n_boxes_abs_padded.padded_size == (100 * 2 * 2, 100 * 4 * 2)

    n_boxes_rel_padded = boxes_abs_padded.rel_pos()
    assert n_boxes_rel_padded.frame_size == None
    assert n_boxes_rel_padded.padded_size == (400, 800)

    real_boxes_abs_padded = boxes_abs_padded.fit_to_padded_size()
    assert real_boxes_abs_padded.padded_size == None
    assert real_boxes_abs_padded.frame_size == (80 * 2 * 2, 80 * 4 * 2)
    assert boxes_equal(real_boxes_abs_padded, boxes_abs)

    real_boxes_rel_padded = boxes_rel_padded.fit_to_padded_size()
    assert real_boxes_rel_padded.padded_size == None
    assert real_boxes_rel_padded.frame_size == None
    assert not boxes_equal(real_boxes_rel_padded, boxes_abs)


def test_boxes_slice():
    boxes = waymo_dataset.get(TEST_FRAME)["front"].boxes2d["gt_boxes_2d"][0]
    boxes = boxes[0:1]
    assert boxes.shape[0] == 1

    frame = waymo_dataset.get(TEST_FRAME)["front"]
    frame = frame.batch()

    assert len(frame.boxes2d["gt_boxes_2d"]) == 1
    assert len(frame.boxes2d["gt_boxes_2d"][0]) == 2
    assert len(frame.boxes2d["gt_boxes_2d"][0][0].shape) == 2

    # Supposed to get 3th time the number of boxes on the temproal dimension
    frame = torch.cat([frame, frame, frame], dim=1)
    assert len(frame.boxes2d["gt_boxes_2d"]) == 1
    assert len(frame.boxes2d["gt_boxes_2d"][0]) == 2 * 3
    assert len(frame.boxes2d["gt_boxes_2d"][0][0].shape) == 2

    n_frame = frame[0:1, 2:6]
    assert len(n_frame.boxes2d["gt_boxes_2d"]) == 1
    assert len(n_frame.boxes2d["gt_boxes_2d"][0]) == 4
    assert len(n_frame.boxes2d["gt_boxes_2d"][0][0].shape) == 2

    n_frame = frame[0, 2:6]
    assert len(n_frame.boxes2d["gt_boxes_2d"]) == 4
    assert len(n_frame.boxes2d["gt_boxes_2d"][0].shape) == 2

    n_frame = frame[0:1, 0]
    assert len(n_frame.boxes2d["gt_boxes_2d"]) == 1
    assert len(n_frame.boxes2d["gt_boxes_2d"][0].shape) == 2

    n_frame = frame[0, 0]
    assert len(n_frame.boxes2d["gt_boxes_2d"].shape) == 2


def test_crop_abs():
    image = np.zeros((3, 843, 1500))
    boxes = [[298, 105, 50, 50], [1250, 105, 50, 50], [298, 705, 50, 50], [1250, 705, 50, 50]]
    frame = aloscene.Frame(image)
    labels = aloscene.Labels([0, 0, 0, 0], labels_names=["boxes"])
    boxes = aloscene.BoundingBoxes2D(
        boxes, boxes_format="xcyc", frame_size=(frame.H, frame.W), absolute=True, labels=labels
    )
    frame.append_boxes2d(boxes)
    frame = frame.crop(H_crop=(0.0, 0.4), W_crop=(0.0, 0.4))
    assert torch.allclose(frame.boxes2d[0].as_tensor(), boxes[0].as_tensor())
    assert np.allclose(frame.boxes2d.frame_size[0], frame.HW[0])
    assert np.allclose(frame.boxes2d.frame_size[1], frame.HW[1])


if __name__ == "__main__":
    test_boxes_from_dt()
    test_boxes_rel_xcyc()
    test_boxes_rel_xcyc()
    test_boxes_rel_xyxy()
    test_boxes_abs_xcyc()
    test_boxes_abs_yxyx()
    test_boxes_abs_xyxy()
    # test_padded_boxes() Outdated
    test_boxes_slice()
    test_crop_abs()
