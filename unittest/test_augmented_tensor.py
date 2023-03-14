import pytest
import aloscene
import torch
from aloscene import Frame, Disparity, Flow

def test_batch_list_intersection_property():
    """
    When batching with intersection flag, a property should be set to None if:
    - it is not set in every tensor
    - or it does not have same value in every tensor
    """
    # case #1: one tensor has None
    f1 = Frame(torch.ones(3,10,10))
    f2 = Frame(torch.ones(3,10,10))
    f1.baseline = 0.5
    f2.baseline = None
    f = aloscene.batch_list([f1, f2], intersection=True)
    assert f.baseline is None
    with pytest.raises(RuntimeError):
        f = aloscene.batch_list([f1, f2], intersection=False) # old behavior was to raise a RuntimeError

    # case #2: tensors have different values
    f2.baseline = 1
    f = aloscene.batch_list([f1, f2], intersection=True)
    assert f.baseline is None
    with pytest.raises(RuntimeError):
        f = aloscene.batch_list([f1, f2], intersection=False) # old behavior was to raise a RuntimeError

def test_batch_list_intersection_mergeable_child():
    """
    When batching with intersection flag, an unmergeable child should be set to None if:
    - it is not set in every tensor
    """
    # case #1: one tensor has None
    f1 = Frame(torch.ones(3,10,10))
    f2 = Frame(torch.ones(3,10,10))
    f1.append_disparity(Disparity(torch.ones(1,10,10)))
    f = aloscene.batch_list([f1, f2], intersection=True)
    assert f.disparity is None
    with pytest.raises(TypeError):
        f = aloscene.batch_list([f1, f2], intersection=False) # old behavior was to raise a TypeError

    # case #2: both tensors have disparity
    f2.append_disparity(Disparity(torch.ones(1,10,10)))
    f = aloscene.batch_list([f1, f2], intersection=True)
    assert f.disparity.shape == (2, 1, 10, 10)
    f = aloscene.batch_list([f1, f2], intersection=False) # old behavior was the same : merge the  disparities
    assert f.disparity.shape == (2, 1, 10, 10)

def test_batch_list_intersection_unmergeable_child():
    """
    When batching with intersection flag, an unmergeable child should always be stacked in a list
    """
    f1 = Frame(torch.ones(3,10,10))
    f2 = Frame(torch.ones(3,10,10))
    f1.append_flow(Flow(torch.ones(2,10,10)))
    f = aloscene.batch_list([f1, f2], intersection=True)
    assert isinstance(f.flow, list)
    assert len(f.flow) == 2
    assert f.flow[0].shape == (2, 10, 10)
    assert f.flow[1] is None
    f = aloscene.batch_list([f1, f2], intersection=False) # old behavior was the same : stack children in list
    assert isinstance(f.flow, list)
    assert len(f.flow) == 2
    assert f.flow[0].shape == (2, 10, 10)
    assert f.flow[1] is None

if __name__ == "__main__":
    test_batch_list_intersection_property()
    test_batch_list_intersection_mergeable_child()
    test_batch_list_intersection_unmergeable_child()