import aloscene
from alodataset import transforms as T
import numpy as np
import torch
import random


def test_color_jitter():

    data = np.random.uniform(0, 1, (3, 200, 200))

    frame = aloscene.Frame(data, names=("C", "H", "W"), normalization="01")
    frame_aug = T.ColorJitter()(frame)
    assert not torch.allclose(frame.as_tensor(), frame_aug.as_tensor())

    # Testing temporal
    frame1 = frame.temporal()
    frame2 = frame.temporal()
    assert torch.allclose(frame1.as_tensor(), frame2.as_tensor())
    frames = torch.cat([frame1, frame2], dim=0)
    # Different color jitter on sequence
    frame_aug = T.ColorJitter(same_on_sequence=False)(frames)
    assert not torch.allclose(frames.as_tensor(), frame_aug.as_tensor())
    assert not torch.allclose(frame_aug[0].as_tensor(), frame_aug[1].as_tensor())
    # Same color jitter on sequence
    frame_aug = T.ColorJitter(same_on_sequence=True)(frames)
    assert torch.allclose(frame_aug[0].as_tensor(), frame_aug[1].as_tensor())

    # Testing a frame set without sequence
    frame_set = {"frame1": frame.clone(), "frame2": frame.clone()}
    assert torch.allclose(frame_set["frame1"].as_tensor(), frame_set["frame2"].as_tensor())
    # Different color jitter on the set of image
    n_frame_set = T.ColorJitter(same_on_frames=False)(frame_set)
    assert not torch.allclose(n_frame_set["frame1"].as_tensor(), n_frame_set["frame2"].as_tensor())
    # Same color jitter on the image set
    n_frame_set = T.ColorJitter(same_on_frames=True)(frame_set)
    assert torch.allclose(n_frame_set["frame1"].as_tensor(), n_frame_set["frame2"].as_tensor())

    # Testing a frame set with augmentations
    frame_set = {
        "frame1": torch.cat([frame.temporal(), frame.temporal()], dim=0),
        "frame2": torch.cat([frame.temporal(), frame.temporal()], dim=0),
    }
    assert torch.allclose(frame_set["frame1"].as_tensor(), frame_set["frame2"].as_tensor())
    # Different color jitter on the set of image
    n_frame_set = T.ColorJitter(same_on_frames=False, same_on_sequence=False)(frame_set)
    assert not torch.allclose(frame_set["frame1"].as_tensor(), n_frame_set["frame1"].as_tensor())
    assert not torch.allclose(frame_set["frame2"].as_tensor(), n_frame_set["frame2"].as_tensor())
    assert not torch.allclose(n_frame_set["frame1"].as_tensor(), n_frame_set["frame2"].as_tensor())
    assert not torch.allclose(n_frame_set["frame1"][0].as_tensor(), n_frame_set["frame2"][0].as_tensor())
    assert not torch.allclose(n_frame_set["frame1"][1].as_tensor(), n_frame_set["frame2"][1].as_tensor())
    assert not torch.allclose(n_frame_set["frame1"][0].as_tensor(), n_frame_set["frame1"][1].as_tensor())

    # Same color jitter on the set of image but different for each sequence element
    n_frame_set = T.ColorJitter(same_on_frames=True, same_on_sequence=False)(frame_set)
    assert not torch.allclose(frame_set["frame1"].as_tensor(), n_frame_set["frame1"].as_tensor())
    assert not torch.allclose(frame_set["frame2"].as_tensor(), n_frame_set["frame2"].as_tensor())
    assert torch.allclose(n_frame_set["frame1"].as_tensor(), n_frame_set["frame2"].as_tensor())
    assert torch.allclose(n_frame_set["frame1"][0].as_tensor(), n_frame_set["frame2"][0].as_tensor())
    assert torch.allclose(n_frame_set["frame1"][1].as_tensor(), n_frame_set["frame2"][1].as_tensor())
    assert not torch.allclose(n_frame_set["frame1"][0].as_tensor(), n_frame_set["frame1"][1].as_tensor())

    n_frame_set = T.ColorJitter(same_on_frames=True, same_on_sequence=True)(frame_set)
    assert not torch.allclose(frame_set["frame1"].as_tensor(), n_frame_set["frame1"].as_tensor())
    assert not torch.allclose(frame_set["frame2"].as_tensor(), n_frame_set["frame2"].as_tensor())
    assert torch.allclose(n_frame_set["frame1"].as_tensor(), n_frame_set["frame2"].as_tensor())
    assert torch.allclose(n_frame_set["frame1"][0].as_tensor(), n_frame_set["frame2"][0].as_tensor())
    assert torch.allclose(n_frame_set["frame1"][1].as_tensor(), n_frame_set["frame2"][1].as_tensor())
    assert torch.allclose(n_frame_set["frame1"][0].as_tensor(), n_frame_set["frame1"][1].as_tensor())

    # Same color jitter on the set of image but different for each sequence element
    n_frame_set = T.ColorJitter(same_on_frames=False, same_on_sequence=True)(frame_set)
    assert not torch.allclose(frame_set["frame1"].as_tensor(), n_frame_set["frame1"].as_tensor())
    assert not torch.allclose(frame_set["frame2"].as_tensor(), n_frame_set["frame2"].as_tensor())
    assert not torch.allclose(n_frame_set["frame1"].as_tensor(), n_frame_set["frame2"].as_tensor())
    assert not torch.allclose(n_frame_set["frame1"][0].as_tensor(), n_frame_set["frame2"][0].as_tensor())
    assert not torch.allclose(n_frame_set["frame1"][1].as_tensor(), n_frame_set["frame2"][1].as_tensor())
    assert torch.allclose(n_frame_set["frame1"][0].as_tensor(), n_frame_set["frame1"][1].as_tensor())


if __name__ == "__main__":
    # seed everything
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    test_color_jitter()
