import numpy as np
import torch
import aloscene
from aloscene import OrientedBoxes2D

device = torch.device("cpu")


def tensor_equal(tensor1, tensor2, threshold=1e-4):
    if (tensor1 - tensor2).abs().mean() < threshold:
        return True
    else:
        return False


def test_same_box():
    # if cuda not avaible
    if not torch.cuda.is_available():
        return
    box1 = OrientedBoxes2D(torch.tensor([[0.0, 0.0, 2.0, 2.0, 0.0]], device=device))
    box2 = OrientedBoxes2D(torch.tensor([[0.0, 0.0, 2.0, 2.0, 0.0]], device=device))
    expected_iou = torch.tensor([1.0], device=device)
    expected_giou = torch.tensor([1.0], device=device)
    giou, iou = box1.rotated_giou_with(box2, ret_iou=True)
    assert tensor_equal(iou, expected_iou)
    assert tensor_equal(giou, expected_giou)


def test_same_edge():
    # if cuda not avaible
    if not torch.cuda.is_available():
        return
    box1 = OrientedBoxes2D(torch.tensor([[0.0, 0.0, 2.0, 2.0, 0.0]], device=device))
    box2 = OrientedBoxes2D(torch.tensor([[2.0, 0.0, 2.0, 2.0, 0.0]], device=device))
    expected_iou = torch.tensor([0.0], device=device)
    expected_giou = torch.tensor([0.0], device=device)
    giou, iou = box1.rotated_giou_with(box2, ret_iou=True)
    iou = box1.rotated_iou_with(box2)
    assert tensor_equal(iou, expected_iou)
    assert tensor_equal(giou, expected_giou)


def test_1():
    # if cuda not avaible
    if not torch.cuda.is_available():
        return
    box1 = OrientedBoxes2D(torch.tensor([[0.0, 0.0, 2.0, 2.0, 0.0]], device=device))
    box2 = OrientedBoxes2D(torch.tensor([[1.0, 0.0, 2.0, 2.0, 0.0]], device=device))
    expected_iou = torch.tensor([1 / 3], device=device)
    expected_giou = torch.tensor([1 / 3], device=device)
    giou, iou = box1.rotated_giou_with(box2, ret_iou=True)
    assert tensor_equal(iou, expected_iou)
    assert tensor_equal(giou, expected_giou)


def test_2():
    # if cuda not avaible
    if not torch.cuda.is_available():
        return
    box1 = OrientedBoxes2D(torch.tensor([[0.0, 0.0, 2.0, 2.0, 0.0]], device=device))
    box2 = OrientedBoxes2D(torch.tensor([[1.0, 1.0, 2.0, 2.0, 0.0]], device=device))
    expected_iou = torch.tensor([1 / 7], device=device)
    expected_giou = torch.tensor([1 / 7 - 2 / 9], device=device)
    giou, iou = box1.rotated_giou_with(box2, ret_iou=True)
    assert tensor_equal(iou, expected_iou)
    assert tensor_equal(giou, expected_giou)


def test_3():
    # if cuda not avaible
    if not torch.cuda.is_available():
        return
    box1 = OrientedBoxes2D(torch.tensor([[0.0, 0.0, 2.0, 2.0, 0.0]], device=device))
    box2 = OrientedBoxes2D(torch.tensor([[1.0, 1.0, 2.0, 2.0, np.pi / 2]], device=device))
    expected_iou = torch.tensor([1 / 7], device=device)
    expected_giou = torch.tensor([1 / 7 - 2 / 9], device=device)
    giou, iou = box1.rotated_giou_with(box2, ret_iou=True)
    assert tensor_equal(iou, expected_iou)
    assert tensor_equal(giou, expected_giou)


def test_4():
    # if cuda not avaible
    if not torch.cuda.is_available():
        return
    box1 = OrientedBoxes2D(torch.tensor([[0.0, 0.0, 2.0, 2.0, 0.0]], device=device))
    box2 = OrientedBoxes2D(
        torch.tensor([[1.0, 1.0, np.sqrt(2), np.sqrt(2), np.pi / 4]], device=device, dtype=torch.float)
    )
    expected_iou = torch.tensor([0.5 / 5.5], device=device)
    expected_giou = torch.tensor([0.5 / 5.5 - 3.5 / 9], device=device)
    giou, iou = box1.rotated_giou_with(box2, ret_iou=True)
    assert tensor_equal(iou, expected_iou)
    assert tensor_equal(giou, expected_giou)


if __name__ == "__main__":
    test_same_box()
    test_same_edge()
    test_1()
    test_2()
    test_3()
    test_4()
