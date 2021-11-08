from aloscene.labels import Labels
from aloscene.bounding_boxes_2d import BoundingBoxes2D

# from aloscene.renderer import View
from alodataset import WaymoDataset, Split
import aloscene
import torch
import numpy as np

waymo_dataset = WaymoDataset(sample=True)
# waymo_dataset = WaymoDataset(split=Split.VAL, labels=["gt_boxes_2d", "gt_boxes_3d"], sequence_size=2)

TEST_FRAME = 0


def get_frame():
    n_frame = waymo_dataset.get(TEST_FRAME)["front"]
    # Prevent a known issue related to the camera extrinsic parameters
    n_frame.cam_extrinsic = None
    n_frame.cam_intrinsic = None
    n_frame.boxes3d = None
    return n_frame


def tensor_equal(frame1, frame2):
    frame1.rename_(None)
    frame2.rename_(None)
    return torch.allclose(frame1, frame2, atol=1e-4)


def test_frame_from_dt():
    # Go through the loaded sequence
    for data in waymo_dataset.stream_loader():
        assert data["front"].shape[:2] == (2, 3)
        assert data["front"].names == ("T", "C", "H", "W")
        # It should be instance of list since the dataset return a sequence and the
        # boxes are not aligned on T.
        assert isinstance(data["front"].boxes2d["gt_boxes_2d"], list)
        break


def test_frame_01():
    frame = get_frame()
    frame = frame.norm01()

    frame_01 = frame.norm01()
    assert frame_01.normalization == "01"
    frame_255 = frame_01.norm255()
    assert frame_255.normalization == "255"
    frame_resnet = frame.norm_resnet()
    assert frame_resnet.normalization == "resnet"
    my_frame = frame.mean_std_norm((0.42, 0.4, 0.40), (0.41, 0.2, 0.45), name="my_norm")
    assert my_frame.normalization == "my_norm"
    frame_minmax_sym = frame.norm_minmax_sym()
    assert frame_minmax_sym.normalization == "minmax_sym"
    # Assert back equality
    assert tensor_equal(frame_01.norm01(), frame)
    assert tensor_equal(frame_255.norm01(), frame)
    assert tensor_equal(frame_resnet.norm01(), frame)
    assert tensor_equal(my_frame.norm01(), frame)
    assert tensor_equal(frame_minmax_sym.norm01(), frame)


def test_frame_255():
    frame = get_frame()
    frame = frame.norm255()

    frame_01 = frame.norm01()
    assert frame_01.normalization == "01"
    frame_255 = frame_01.norm255()
    assert frame_255.normalization == "255"
    frame_resnet = frame.norm_resnet()
    assert frame_resnet.normalization == "resnet"
    my_frame = frame.mean_std_norm((0.42, 0.4, 0.40), (0.41, 0.2, 0.45), name="my_norm")
    assert my_frame.normalization == "my_norm"
    frame_minmax_sym = frame.norm_minmax_sym()
    assert frame_minmax_sym.normalization == "minmax_sym"
    # Assert back equality
    assert tensor_equal(frame_01.norm255(), frame)
    assert tensor_equal(frame_255.norm255(), frame)

    assert tensor_equal(frame_resnet.norm255(), frame)
    assert tensor_equal(my_frame.norm255(), frame)
    assert tensor_equal(frame_minmax_sym.norm255(), frame)


def test_frame_resnet():
    frame = get_frame()
    frame = frame.norm_resnet()

    frame_01 = frame.norm01()
    assert frame_01.normalization == "01"
    frame_255 = frame_01.norm255()
    assert frame_255.normalization == "255"
    frame_resnet = frame.norm_resnet()
    assert frame_resnet.normalization == "resnet"
    my_frame = frame.mean_std_norm((0.42, 0.4, 0.40), (0.41, 0.2, 0.45), name="my_norm")
    assert my_frame.normalization == "my_norm"
    # Assert back equality
    assert tensor_equal(frame_01.norm_resnet(), frame)
    assert tensor_equal(frame_255.norm_resnet(), frame)
    assert tensor_equal(frame_resnet.norm_resnet(), frame)
    assert tensor_equal(my_frame.norm_resnet(), frame)


def test_frame_minmax_sym():
    frame = get_frame()
    frame = frame.norm_minmax_sym()

    frame_01 = frame.norm01()
    assert frame_01.normalization == "01"
    frame_255 = frame_01.norm255()
    assert frame_255.normalization == "255"
    # Assert back equality
    assert tensor_equal(frame_01.norm_minmax_sym(), frame)
    assert tensor_equal(frame_255.norm_minmax_sym(), frame)


def test_frame_concat():
    frame = get_frame()
    assert len(frame.shape) == 4
    assert frame.names == ("T", "C", "H", "W")

    frame_t0 = frame[0]
    frame_t1 = frame[1]
    assert len(frame_t0.shape) == 3
    assert frame_t0.names == ("C", "H", "W")
    assert len(frame_t1.shape) == 3
    assert frame_t1.names == ("C", "H", "W")

    n_frame_t0 = frame_t0.temporal()
    n_frame_t1 = frame_t1.temporal()
    assert len(n_frame_t0.shape) == 4
    assert n_frame_t0.names == ("T", "C", "H", "W")
    assert len(n_frame_t1.shape) == 4
    assert n_frame_t1.names == ("T", "C", "H", "W")

    frames = torch.cat([n_frame_t0, n_frame_t1], dim=0)
    assert len(frames.shape) == 4
    assert frames.names == ("T", "C", "H", "W")


def test_frame_crop():
    frame = get_frame()
    cropped_frame = frame[0, :, 400:900, 500:800]
    assert cropped_frame.shape == (3, 500, 300)


def test_flip():
    frame = get_frame()
    # Horizontal flip (Only api test for now)
    cropped_frame_hflip = frame.hflip()


def test_batch_temporal_frame():
    frames = waymo_dataset.get(TEST_FRAME)["front"]

    # Check temporality
    assert len(frames.shape) == 4
    assert frames.names == ("T", "C", "H", "W")

    gt_boxes_2d = frames.boxes2d["gt_boxes_2d"]
    assert isinstance(gt_boxes_2d, list)
    assert len(gt_boxes_2d) == 2
    boxes_type = type(gt_boxes_2d[0])

    # ####
    # Expand with batch dimension
    # #####
    batch_frames = frames.batch()
    assert len(batch_frames.shape) == 5
    assert batch_frames.names == ("B", "T", "C", "H", "W")
    # Check the new label structure
    gt_boxes_2d = batch_frames.boxes2d["gt_boxes_2d"]
    assert isinstance(gt_boxes_2d, list)
    assert len(gt_boxes_2d) == 1  # The new batch dimension
    assert len(gt_boxes_2d[0]) == 2  # The temporal dimension
    # Check that the structure of the previous frames are still correct
    gt_boxes_2d = frames.boxes2d["gt_boxes_2d"]
    assert isinstance(gt_boxes_2d, list)
    assert len(gt_boxes_2d) == 2

    # ####
    # Slice on the temporal dimension
    # #####
    batch_frames_t0 = batch_frames[:, 0, ::]
    gt_boxes_2d = batch_frames_t0.boxes2d["gt_boxes_2d"]

    # Check the new label structure
    assert isinstance(gt_boxes_2d, list)
    assert len(gt_boxes_2d) == 1  # The new batch dimension
    assert type(gt_boxes_2d[0]) == boxes_type
    # Check that the structure of the previous frames are still correct
    gt_boxes_2d = batch_frames.boxes2d["gt_boxes_2d"]
    assert isinstance(gt_boxes_2d, list)
    assert len(gt_boxes_2d) == 1  # The new batch dimension
    assert len(gt_boxes_2d[0]) == 2  # The temporal dimension

    # ####
    # Slice on the batch dimension
    # #####
    frames = batch_frames[0]
    gt_boxes_2d = frames.boxes2d["gt_boxes_2d"]
    # Check the new label structure
    assert isinstance(gt_boxes_2d, list)
    assert len(gt_boxes_2d) == 2  # The new batch dimension
    assert type(gt_boxes_2d[0]) == boxes_type
    # Check that the structure of the previous frames are still correct
    gt_boxes_2d = batch_frames.boxes2d["gt_boxes_2d"]
    assert isinstance(gt_boxes_2d, list)
    assert len(gt_boxes_2d) == 1  # The new batch dimension
    assert len(gt_boxes_2d[0]) == 2  # The temporal dimension

    ####
    # Concat along a the batch dimension
    ####
    batch_frames = torch.cat([batch_frames, batch_frames, batch_frames], dim=0)
    assert len(batch_frames.shape) == 5
    assert batch_frames.shape[0] == 3
    assert batch_frames.shape[1] == 2
    assert batch_frames.names == ("B", "T", "C", "H", "W")
    # Check the new label structure
    gt_boxes_2d = batch_frames.boxes2d["gt_boxes_2d"]
    assert isinstance(gt_boxes_2d, list)
    assert len(gt_boxes_2d) == 3  # The new batch dimension
    assert len(gt_boxes_2d[0]) == 2  # The temporal dimension

    ####
    # Merge on the temporal dimension
    ####
    extended_temporality_frames = torch.cat([batch_frames, batch_frames, batch_frames], dim=1)
    assert len(extended_temporality_frames.shape) == 5
    assert extended_temporality_frames.shape[0] == 3
    assert extended_temporality_frames.shape[1] == 6
    assert extended_temporality_frames.names == ("B", "T", "C", "H", "W")
    # Check the new label structure
    gt_boxes_2d = extended_temporality_frames.boxes2d["gt_boxes_2d"]
    assert isinstance(gt_boxes_2d, list)
    assert len(gt_boxes_2d) == 3  # The new batch dimension
    assert len(gt_boxes_2d[0]) == 6  # The temporal dimension


def test_batch_list_frame():
    frames01 = waymo_dataset.get(TEST_FRAME)["front"]
    frames02 = frames01.hflip()

    assert frames01.names == ("T", "C", "H", "W")
    assert frames02.names == ("T", "C", "H", "W")
    gt_boxes_2d = frames01.boxes2d["gt_boxes_2d"]
    assert isinstance(gt_boxes_2d, list)
    assert len(gt_boxes_2d) == frames01.shape[0]
    gt_boxes_2d = frames02.boxes2d["gt_boxes_2d"]
    assert isinstance(gt_boxes_2d, list)
    assert len(gt_boxes_2d) == frames02.shape[0]

    frames = frames01.batch_list([frames01, frames02])
    assert frames.names == ("B", "T", "C", "H", "W")
    gt_boxes_2d = frames.boxes2d["gt_boxes_2d"]
    assert isinstance(gt_boxes_2d, list)
    assert len(gt_boxes_2d) == frames.shape[0]
    assert len(gt_boxes_2d[0]) == frames.shape[1]

    assert tensor_equal(frames.boxes2d["gt_boxes_2d"][0][0], frames01.boxes2d["gt_boxes_2d"][0])
    assert tensor_equal(frames.boxes2d["gt_boxes_2d"][0][1], frames01.boxes2d["gt_boxes_2d"][1])
    assert tensor_equal(frames.boxes2d["gt_boxes_2d"][1][0], frames02.boxes2d["gt_boxes_2d"][0])
    assert tensor_equal(frames.boxes2d["gt_boxes_2d"][1][1], frames02.boxes2d["gt_boxes_2d"][1])

    # TODO: TO FIXE
    # frames = frames.hflip()
    # assert not tensor_equal(frames.boxes2d["gt_boxes_2d"][0][0], frames01.boxes2d["gt_boxes_2d"][0])
    # assert not tensor_equal(frames.boxes2d["gt_boxes_2d"][0][1], frames01.boxes2d["gt_boxes_2d"][1])
    # assert not tensor_equal(frames.boxes2d["gt_boxes_2d"][1][0], frames02.boxes2d["gt_boxes_2d"][0])
    # assert not tensor_equal(frames.boxes2d["gt_boxes_2d"][1][1], frames02.boxes2d["gt_boxes_2d"][1])


def test_batch_list_errors():
    # no assert but this code should not produce an error
    try:
        H, W = 600, 550
        frame0 = aloscene.Frame(np.random.uniform(0, 1, (3, H, W)), names=("C", "H", "W"))
        disp_mask0 = aloscene.Mask(torch.zeros((1, H, W)), names=("C", "H", "W"))
        disp0 = aloscene.Disparity(torch.zeros((1, H, W)), names=("C", "H", "W"), occlusion=disp_mask0)
        frame0.append_disparity(disp0)
        frame1 = frame0.clone()
        frames = aloscene.Frame.batch_list([frame0, frame1])
    except TypeError:
        assert False, "TypeError during Frame.batch_list()"


def test_recusrive_temporal_batch():

    S = 600

    frame = aloscene.Frame(np.random.uniform(0, 1, (3, 600, 600)), names=("C", "H", "W"))

    boxes1 = BoundingBoxes2D(
        np.array([[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.1, 0.1]]), boxes_format="xcyc", absolute=False, names=("N", None)
    )
    boxes2 = BoundingBoxes2D(
        np.array([[0.3, 0.3, 0.5, 0.5], [0.2, 0.2, 0.1, 0.1]]), boxes_format="xcyc", absolute=False, names=("N", None)
    )

    frame.append_boxes2d(boxes1, "boxes1")
    frame.append_boxes2d(boxes2, "boxes2")

    disp_mask1 = aloscene.Mask(torch.zeros((1, frame.H, frame.W)), names=("C", "H", "W"))
    disp_mask1[:, :, :300] = 1
    disp_mask2 = aloscene.Mask(torch.zeros((1, frame.H, frame.W)), names=("C", "H", "W"))
    disp_mask2[:, :, 300:] = 1

    flow_mask = aloscene.Mask(torch.zeros((1, frame.H, frame.W)), names=("C", "H", "W"))
    flow_mask[:, :, :400] = 1

    flow = aloscene.Flow(torch.zeros((2, frame.H, frame.W)), names=("C", "H", "W"), occlusion=flow_mask)
    disp = aloscene.Disparity(torch.zeros((1, frame.H, frame.W)), names=("C", "H", "W"))
    disp.append_occlusion(disp_mask1, "occ1")
    disp.append_occlusion(disp_mask2, "occ2")

    frame.append_disparity(disp)
    frame.append_flow(flow)

    frame_batch = frame.batch()
    assert frame_batch.shape == (1, 3, S, S)
    assert frame_batch.boxes2d["boxes1"][0].shape == (2, 4)
    assert frame_batch.boxes2d["boxes2"][0].shape == (2, 4)
    assert frame_batch.disparity.shape == (1, 1, S, S)
    assert frame_batch.flow[0].shape == (2, S, S)
    assert frame_batch.disparity.occlusion["occ1"].shape == (1, 1, S, S)
    assert frame_batch.disparity.occlusion["occ2"].shape == (1, 1, S, S)
    assert frame_batch.flow[0].occlusion.shape == (1, S, S)

    frame_batch = torch.cat([frame_batch, frame_batch], dim=0)
    assert frame_batch.shape == (2, 3, S, S)
    assert frame_batch.boxes2d["boxes1"][0].shape == (2, 4)
    assert frame_batch.boxes2d["boxes2"][0].shape == (2, 4)
    assert frame_batch.disparity.shape == (2, 1, S, S)
    assert frame_batch.flow[0].shape == (2, S, S)
    assert frame_batch.disparity.occlusion["occ1"].shape == (2, 1, S, S)
    assert frame_batch.disparity.occlusion["occ2"].shape == (2, 1, S, S)
    assert frame_batch.flow[0].occlusion.shape == (1, S, S)

    frame = frame_batch[0]
    assert frame.shape == (3, S, S)
    assert frame.boxes2d["boxes1"].shape == (2, 4)
    assert frame.boxes2d["boxes2"].shape == (2, 4)
    assert frame.disparity.shape == (1, S, S)
    assert frame.flow.shape == (2, S, S)
    assert frame.disparity.occlusion["occ1"].shape == (1, S, S)
    assert frame.disparity.occlusion["occ2"].shape == (1, S, S)
    assert frame.flow.occlusion.shape == (1, S, S)

    frame_temporal = frame.temporal()
    assert frame_temporal.shape == (1, 3, S, S)
    assert frame_temporal.boxes2d["boxes1"][0].shape == (2, 4)
    assert frame_temporal.boxes2d["boxes2"][0].shape == (2, 4)
    assert frame_temporal.disparity.shape == (1, 1, S, S)
    assert frame_temporal.flow[0].shape == (2, S, S)
    assert frame_temporal.disparity.occlusion["occ1"].shape == (1, 1, S, S)
    assert frame_temporal.disparity.occlusion["occ2"].shape == (1, 1, S, S)
    assert frame_temporal.flow[0].occlusion.shape == (1, S, S)

    frame_temporal = torch.cat([frame_temporal, frame_temporal], dim=0)
    assert frame_temporal.shape == (2, 3, S, S)
    assert frame_temporal.boxes2d["boxes1"][0].shape == (2, 4)
    assert frame_temporal.boxes2d["boxes2"][0].shape == (2, 4)
    assert frame_temporal.disparity.shape == (2, 1, S, S)
    assert frame_temporal.flow[0].shape == (2, S, S)
    assert frame_temporal.disparity.occlusion["occ1"].shape == (2, 1, S, S)
    assert frame_temporal.disparity.occlusion["occ2"].shape == (2, 1, S, S)
    assert frame_temporal.flow[0].occlusion.shape == (1, S, S)

    frame = frame_temporal[0]
    assert frame.shape == (3, S, S)
    assert frame.boxes2d["boxes1"].shape == (2, 4)
    assert frame.boxes2d["boxes2"].shape == (2, 4)
    assert frame.disparity.shape == (1, S, S)
    assert frame.flow.shape == (2, S, S)
    assert frame.disparity.occlusion["occ1"].shape == (1, S, S)
    assert frame.disparity.occlusion["occ2"].shape == (1, S, S)
    assert frame.flow.occlusion.shape == (1, S, S)

    frame_temporal_batch = frame.temporal().batch()
    assert frame_temporal_batch.shape == (1, 1, 3, S, S)
    assert frame_temporal_batch.boxes2d["boxes1"][0][0].shape == (2, 4)
    assert frame_temporal_batch.boxes2d["boxes2"][0][0].shape == (2, 4)
    assert frame_temporal_batch.disparity.shape == (1, 1, 1, S, S)
    assert frame_temporal_batch.flow[0][0].shape == (2, S, S)
    assert frame_temporal_batch.disparity.occlusion["occ1"].shape == (1, 1, 1, S, S)
    assert frame_temporal_batch.disparity.occlusion["occ2"].shape == (1, 1, 1, S, S)
    assert frame_temporal_batch.flow[0][0].occlusion.shape == (1, S, S)

    frame_temporals_batch = torch.cat([frame_temporal_batch, frame_temporal_batch], dim=1)
    assert frame_temporals_batch.shape == (1, 2, 3, S, S)
    assert frame_temporals_batch.boxes2d["boxes1"][0][1].shape == (2, 4)
    assert frame_temporals_batch.boxes2d["boxes2"][0][1].shape == (2, 4)
    assert frame_temporals_batch.disparity.shape == (1, 2, 1, S, S)
    assert frame_temporals_batch.flow[0][1].shape == (2, S, S)
    assert frame_temporals_batch.disparity.occlusion["occ1"].shape == (1, 2, 1, S, S)
    assert frame_temporals_batch.disparity.occlusion["occ2"].shape == (1, 2, 1, S, S)
    assert frame_temporals_batch.flow[0][1].occlusion.shape == (1, S, S)

    frame_temporal_batchs = torch.cat([frame_temporal_batch, frame_temporal_batch], dim=0)
    assert frame_temporal_batchs.shape == (2, 1, 3, S, S)
    assert frame_temporal_batchs.boxes2d["boxes1"][1][0].shape == (2, 4)
    assert frame_temporal_batchs.boxes2d["boxes2"][1][0].shape == (2, 4)
    assert frame_temporal_batchs.disparity.shape == (2, 1, 1, S, S)
    assert frame_temporal_batchs.flow[1][0].shape == (2, S, S)
    assert frame_temporal_batchs.disparity.occlusion["occ1"].shape == (2, 1, 1, S, S)
    assert frame_temporal_batchs.disparity.occlusion["occ2"].shape == (2, 1, 1, S, S)
    assert frame_temporal_batchs.flow[1][0].occlusion.shape == (1, S, S)


def test_recursive_temporal_batch_bis():
    H, W = 600, 550
    frame = aloscene.Frame(np.random.uniform(0, 1, (3, H, W)), names=("C", "H", "W"))
    disp_mask = aloscene.Mask(torch.zeros((1, H, W)), names=("C", "H", "W"))
    disp = aloscene.Disparity(torch.zeros((1, H, W)), names=("C", "H", "W"), occlusion=disp_mask)
    frame.append_disparity(disp)

    frame_batch = frame.batch()
    assert frame_batch.names == ("B", "C", "H", "W")
    assert frame_batch.shape == (1, 3, H, W)
    assert frame_batch.disparity.names == ("B", "C", "H", "W")
    assert frame_batch.disparity.shape == (1, 1, H, W)
    assert frame_batch.disparity.occlusion.names == ("B", "C", "H", "W")
    assert frame_batch.disparity.occlusion.shape == (1, 1, H, W)

    assert frame.names == ("C", "H", "W")
    assert frame.shape == (3, H, W)
    assert frame.disparity.names == ("C", "H", "W")
    assert frame.disparity.shape == (1, H, W)
    assert frame.disparity.occlusion.names == ("C", "H", "W")
    assert frame.disparity.occlusion.shape == (1, H, W)

    frame_temporal = frame.temporal()
    assert frame_temporal.names == ("T", "C", "H", "W")
    assert frame_temporal.shape == (1, 3, H, W)
    assert frame_temporal.disparity.names == ("T", "C", "H", "W")
    assert frame_temporal.disparity.shape == (1, 1, H, W)
    assert frame_temporal.disparity.occlusion.names == ("T", "C", "H", "W")
    assert frame_temporal.disparity.occlusion.shape == (1, 1, H, W)

    frame_temporal_batch = frame.temporal().batch()
    assert frame_temporal_batch.names == ("B", "T", "C", "H", "W")
    assert frame_temporal_batch.shape == (1, 1, 3, H, W)
    assert frame_temporal_batch.disparity.names == ("B", "T", "C", "H", "W")
    assert frame_temporal_batch.disparity.shape == (1, 1, 1, H, W)
    assert frame_temporal_batch.disparity.occlusion.names == ("B", "T", "C", "H", "W")
    assert frame_temporal_batch.disparity.occlusion.shape == (1, 1, 1, H, W)


def test_pad():
    S = 600
    frame = aloscene.Frame(np.random.uniform(0, 1, (3, 600, 600)), normalization="01", names=("C", "H", "W"))
    frame = frame.norm_resnet()
    boxes1 = BoundingBoxes2D(
        np.array([[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.1, 0.1]]), boxes_format="xcyc", absolute=False, names=("N", None)
    )
    boxes2 = BoundingBoxes2D(
        np.array([[0.3, 0.3, 0.5, 0.5], [0.2, 0.2, 0.1, 0.1]]), boxes_format="xcyc", absolute=False, names=("N", None)
    )
    frame.append_boxes2d(boxes1, "boxes1")
    frame.append_boxes2d(boxes2, "boxes2")
    n_frame = frame.pad(offset_y=(0.0, 0.1), offset_x=(0.0, 0.1))


def test_device_propagation():

    S = 600
    frame = aloscene.Frame(np.random.uniform(0, 1, (3, 600, 600)), names=("C", "H", "W"))

    boxes1 = BoundingBoxes2D(
        np.array([[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.1, 0.1]]), boxes_format="xcyc", absolute=False, names=("N", None)
    )
    boxes2 = BoundingBoxes2D(
        np.array([[0.3, 0.3, 0.5, 0.5], [0.2, 0.2, 0.1, 0.1]]), boxes_format="xcyc", absolute=False, names=("N", None)
    )

    frame.append_boxes2d(boxes1, "boxes1")
    frame.append_boxes2d(boxes2, "boxes2")

    disp_mask1 = aloscene.Mask(torch.zeros((1, frame.H, frame.W)), names=("C", "H", "W"))
    disp_mask1[:, :, :300] = 1
    disp_mask2 = aloscene.Mask(torch.zeros((1, frame.H, frame.W)), names=("C", "H", "W"))
    disp_mask2[:, :, 300:] = 1

    flow_mask = aloscene.Mask(torch.zeros((1, frame.H, frame.W)), names=("C", "H", "W"))
    flow_mask[:, :, :400] = 1

    flow = aloscene.Flow(torch.zeros((2, frame.H, frame.W)), names=("C", "H", "W"), occlusion=flow_mask)
    disp = aloscene.Disparity(torch.zeros((1, frame.H, frame.W)), names=("C", "H", "W"))
    disp.append_occlusion(disp_mask1, "occ1")
    disp.append_occlusion(disp_mask2, "occ2")

    frame.append_disparity(disp)
    frame.append_flow(flow)

    assert frame.device == torch.device("cpu")
    assert frame.boxes2d["boxes1"][0].device == torch.device("cpu")
    assert frame.boxes2d["boxes2"][0].device == torch.device("cpu")
    assert frame.disparity.device == torch.device("cpu")
    assert frame.flow[0].device == torch.device("cpu")
    assert frame.disparity.occlusion["occ1"].device == torch.device("cpu")
    assert frame.disparity.occlusion["occ2"].device == torch.device("cpu")
    assert frame.flow[0].occlusion.device == torch.device("cpu")

    cuda_frame = frame.cuda()
    assert cuda_frame.boxes2d["boxes1"][0].device != torch.device("cpu")
    assert cuda_frame.boxes2d["boxes2"][0].device != torch.device("cpu")
    assert cuda_frame.disparity.device != torch.device("cpu")
    assert cuda_frame.flow[0].device != torch.device("cpu")
    assert cuda_frame.disparity.occlusion["occ1"].device != torch.device("cpu")
    assert cuda_frame.disparity.occlusion["occ2"].device != torch.device("cpu")
    assert cuda_frame.flow[0].occlusion.device != torch.device("cpu")

    cpu_frame = frame.cpu()
    assert cpu_frame.boxes2d["boxes1"][0].device == torch.device("cpu")
    assert cpu_frame.boxes2d["boxes2"][0].device == torch.device("cpu")
    assert cpu_frame.disparity.device == torch.device("cpu")
    assert cpu_frame.flow[0].device == torch.device("cpu")
    assert cpu_frame.disparity.occlusion["occ1"].device == torch.device("cpu")
    assert cpu_frame.disparity.occlusion["occ2"].device == torch.device("cpu")
    assert cpu_frame.flow[0].occlusion.device == torch.device("cpu")

    pass


def test_frame_label():
    frame = aloscene.Frame(np.random.uniform(0, 1, (3, 600, 600)), names=("C", "H", "W"))

    label = aloscene.Labels([1], encoding="id")
    frame.append_labels(label)

    assert frame.labels == 1
    frame = frame.batch()
    assert len(frame.labels) == 1
    frame = frame.temporal()
    assert len(frame.labels) == 1 and len(frame.labels[0]) == 1
    frame1 = frame.clone()
    frame2 = frame.clone()
    frames_batch = torch.cat([frame1, frame2], dim=1)
    assert len(frames_batch.labels) == 1 and len(frames_batch.labels[0]) == 2
    frames_temporal = torch.cat([frame1, frame2], dim=0)
    assert len(frames_temporal.labels) == 2 and len(frames_temporal.labels[0]) == 1
    n_frame = frames_batch[:, 0]
    assert len(n_frame.labels) == 1 and len(n_frame.labels[0]) == 1
    n_frame = frames_temporal[0]
    assert len(n_frame.labels) == 1 and len(n_frame.labels[0]) == 1
    frame = n_frame[0]
    assert len(frame.labels) == 1 and len(frame.labels.shape) == 1


if __name__ == "__main__":
    test_frame_label()
    test_frame_from_dt()
    test_frame_01()
    test_frame_255()
    test_frame_resnet()
    test_frame_concat()
    test_frame_crop()
    test_flip()
    test_batch_temporal_frame()
    test_batch_list_frame()
    test_recusrive_temporal_batch()
    test_batch_list_errors()
    test_recursive_temporal_batch_bis()
    test_pad()
    test_device_propagation()
