import torch
import numpy as np
from alodataset import WaymoDataset
from aloscene import BoundingBoxes3D


TEST_FRAME = 0
CAMERAS = ["front", "front_left", "front_right", "side_left", "side_right"]
device = torch.device("cpu")

waymo_dataset = WaymoDataset(sample=True)


def boxes_equal(boxes1, boxes2):
    if torch.mean((boxes1 - boxes2).rename_(None).abs()) < 1e-4:
        return True
    else:
        return False


def tensor_equal(tensor1, tensor2, threshold=1e-4):
    if (tensor1 - tensor2).abs().mean() < threshold:
        return True
    else:
        return False


def test_boxes_from_dt():
    boxes = waymo_dataset.get(TEST_FRAME)["front"].boxes3d["gt_boxes_3d"]
    assert isinstance(boxes, list)
    assert len(boxes) == 2


def test_camera_calib_from_dt():
    frame = waymo_dataset.get(TEST_FRAME)["front"]
    frame = frame[:2]
    cam_intrinsic = frame.cam_intrinsic
    cam_extrinsic = frame.cam_extrinsic
    assert cam_intrinsic.shape[0] == 2 and cam_intrinsic.shape[1] == 3 and cam_intrinsic.shape[2] == 4
    assert cam_extrinsic.shape[0] == 2 and cam_extrinsic.shape[1] == 4 and cam_extrinsic.shape[2] == 4


def test_shape_vertices_3d():
    boxes = waymo_dataset.get(TEST_FRAME)["front"].boxes3d["gt_boxes_3d"][0]
    vertices_3d = BoundingBoxes3D.get_vertices_3d(boxes)
    assert vertices_3d.shape[0] == boxes.shape[0] and vertices_3d.shape[1] == 8 and vertices_3d.shape[2] == 3


def test_shape_vertices_3d_proj():
    boxes = waymo_dataset.get(TEST_FRAME)["front"].boxes3d["gt_boxes_3d"][0]
    cam_intrinsic = waymo_dataset.get(TEST_FRAME)["front"].cam_intrinsic[0]
    cam_extrinsic = waymo_dataset.get(TEST_FRAME)["front"].cam_extrinsic[0]
    vertices_3d_proj = BoundingBoxes3D.get_vertices_3d_proj(boxes, cam_intrinsic, cam_extrinsic)
    assert (
        vertices_3d_proj.shape[0] == boxes.shape[0]
        and vertices_3d_proj.shape[1] == 8
        and vertices_3d_proj.shape[2] == 2
    )


def test_shape_boxes_3d_proj():
    frame = waymo_dataset.get(TEST_FRAME)["front"]
    boxes = waymo_dataset.get(TEST_FRAME)["front"].boxes3d["gt_boxes_3d"][0]
    cam_intrinsic = waymo_dataset.get(TEST_FRAME)["front"].cam_intrinsic[0]
    cam_extrinsic = waymo_dataset.get(TEST_FRAME)["front"].cam_extrinsic[0]
    boxes_proj = BoundingBoxes3D.get_enclosing_box_2d(boxes, cam_intrinsic, cam_extrinsic, frame.HW)
    assert boxes_proj.shape[0] == boxes.shape[0] and boxes_proj.shape[1] == 4


def test_hflip():
    boxes = waymo_dataset.get(0)["front"].boxes3d["gt_boxes_3d"][0]
    cam_extrinsic = waymo_dataset.get(0)["front"].cam_extrinsic[0]
    flipped_boxes = BoundingBoxes3D.boxes_3d_hflip(boxes, cam_extrinsic)

    assert tensor_equal(
        torch.tensor(flipped_boxes.shape, dtype=torch.float32), torch.tensor(boxes.shape, dtype=torch.float32)
    )
    for i in range(len(boxes.names)):
        assert boxes.names[i] == flipped_boxes.names[i]
    boxes = boxes.as_tensor()
    flipped_boxes = flipped_boxes.as_tensor()

    assert tensor_equal(flipped_boxes[:, 3:6], boxes[:, 3:6])
    dist_boxes = torch.norm(boxes[:, :3], dim=-1)
    dist_flipped_boxes = torch.norm(flipped_boxes[:, :3], dim=-1)

    assert tensor_equal(dist_boxes, dist_flipped_boxes, threshold=1e-1)


def test_giou3d_same_box():
    box1 = BoundingBoxes3D(torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0]], device=device))
    giou, iou = box1.giou3d_with(box1, ret_iou3d=True)
    expected_iou = torch.tensor([1.0], device=device)
    expected_giou = torch.tensor([1.0], device=device)
    assert tensor_equal(iou, expected_iou)
    assert tensor_equal(giou, expected_giou)


def test_giou3d_same_face():
    box1 = BoundingBoxes3D(torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0]], device=device))
    box2 = BoundingBoxes3D(torch.tensor([[2.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0]], device=device))
    giou, iou = box1.giou3d_with(box2, ret_iou3d=True)
    expected_iou = torch.tensor([0.0], device=device)
    expected_giou = torch.tensor([0.0], device=device)
    assert tensor_equal(iou, expected_iou)
    assert tensor_equal(giou, expected_giou)


def test_giou3d_1():
    box1 = BoundingBoxes3D(torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0]], device=device))
    box2 = BoundingBoxes3D(torch.tensor([[1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 0.0]], device=device))
    giou, iou = box1.giou3d_with(box2, ret_iou3d=True)
    expected_iou = torch.tensor([1 / 15], device=device)
    expected_giou = torch.tensor([1 / 15 - 12 / 3**3], device=device)
    assert tensor_equal(iou, expected_iou)
    assert tensor_equal(giou, expected_giou)


def test_giou3d_2():
    box1 = BoundingBoxes3D(torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0]], device=device))
    box2 = BoundingBoxes3D(torch.tensor([[1.0, 1.0, 1.0, 2.0, 2.0, 2.0, np.pi / 2]], device=device)).to(torch.float)
    giou, iou = box1.giou3d_with(box2, ret_iou3d=True)
    expected_iou = torch.tensor([1 / 15], device=device)
    expected_giou = torch.tensor([1 / 15 - 12 / 3**3], device=device)
    assert tensor_equal(iou, expected_iou)
    assert tensor_equal(giou, expected_giou)


if __name__ == "__main__":
    test_boxes_from_dt()
    test_camera_calib_from_dt()
    test_shape_vertices_3d()
    test_shape_vertices_3d_proj()
    test_shape_boxes_3d_proj()
    test_hflip()
    # if cuda is available, run the tests on cuda. (giou use custom cuda op to compile)
    if torch.cuda.is_available():
        test_giou3d_same_box()
        test_giou3d_same_face()
        test_giou3d_1()
        test_giou3d_2()
