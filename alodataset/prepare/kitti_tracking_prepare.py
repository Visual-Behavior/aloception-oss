from aloserving.serving_models import EFFRaftServing
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from alodataset import KittiTrackingDataset, Split

"""
Prepare dataset to add depth to kitti_tracking, annotated by RAFT
WARNING:  Not tested
"""

path = "/data/kitti_tracking/training"
serving = EFFRaftServing.from_baseline("EFFRAFT-ITERS12-RGB-368-1216", executor="onnx")
serving.height = 368
serving.width = 1216
warmup = False
if not os.path.isdir(os.path.join(path, "depth_02")):
    os.mkdir(os.path.join(path, "depth_02"))

for split in tqdm((Split.TRAIN, Split.VAL)):
    dataset = KittiTrackingDataset(sequence_size=1, skip=0, sequence_skip=0, split=split)
    for i in tqdm(range(len(dataset))):
        data = dataset.items[i]
        seq = data["sequence"]
        seq_filled = str(seq).zfill(4)
        if not os.path.isdir(os.path.join(path, "depth_02", seq_filled)):
            os.mkdir(os.path.join(path, "depth_02", seq_filled))
        frame_n = str(data["temporal_sequence"][0]).zfill(6)
        frame = dataset[i]
        baseline = frame["left"].baseline
        focal = dataset[1020]["left"].cam_intrinsic[0, 0, 0]
        outputs = serving(
            frame1=frame["left"][0].permute(1, 2, 0).numpy(),
            frame2=frame["right"][0].permute(1, 2, 0).numpy(),
            depth_from_disparity=True,
            baseline=float(baseline),
            focal_length=float(focal) * 1216.0 / frame["left"].W,
        )
        img_path = os.path.join(path, "depth_02", seq_filled, frame_n)
        np.savez_compressed(img_path, (outputs["depth"].value * 100).astype(np.ushort))
