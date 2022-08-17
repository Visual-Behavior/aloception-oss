import os
import torch
import numpy as np

from alodataset import BaseDataset, SplitMixin
from aloscene import Frame, Pose, CameraIntrinsic


class KittiOdometryDataset(BaseDataset, SplitMixin):
    def __init__(self, name="kitti_odometry", sequance=0, grayscale=True, right_frame=True, **kwargs):
        super().__init__(name=name, **kwargs)

        if self.sample:
            return

        datasets = os.path.join(self.dataset_dir)

        # Create intrinsic with file calib.txt
        with open(os.path.join(self.dataset_dir, "sequences", f"{sequance:02d}", "calib.txt"), "r") as file:
            self.calib = file.readlines()
            left_intrinsic = np.array([float(x) for x in self.calib[0].split()[1:]]).reshape(3, 4)
            right_intrinsic = np.array([float(x) for x in self.calib[1].split()[1:]]).reshape(3, 4)
            self.left_intrinsic = CameraIntrinsic(left_intrinsic)
            self.right_intrinsic = CameraIntrinsic(right_intrinsic)

        # Load poses of the sequence
        self.poses = None
        if sequance <= 10:
            with open(os.path.join(datasets, "poses", f"{sequance:02d}.txt"), "r") as file:
                self.poses = file.readlines()

        # Load times of each frame
        with open(os.path.join(self.dataset_dir, "sequences", f"{sequance:02d}", "times.txt"), "r") as file:
            self.times = file.readlines()

        self.folders = {
            "left": os.path.join(datasets, "sequences", f"{sequance:02d}", f"image_{0 if grayscale else 2}"),
            "right": os.path.join(datasets, "sequences", f"{sequance:02d}", f"image_{1 if grayscale else 3}")
            if right_frame
            else None,
        }

    def __len__(self):
        if self.sample:
            return len(self.items)
        return len(self.times)

    def getitem(self, idx: int):
        """
        Loads a single frame from the dataset.

        Parameters
        ----------
        idx : int
            Index of the frame to load.

        Returns
        -------
        Dict[str, Frame]
            Dictionary with the loaded frame and pose.
        """

        if self.sample:
            return BaseDataset.__getitem__(self, idx)

        frames = {
            "left": Frame(os.path.join(self.folders["left"], f"{idx:06d}.png")),
        }

        frames["left"].baseline = 0.54
        frames["left"].append_cam_intrinsic(self.left_intrinsic)
        frames["left"].timestamp = float(self.times[idx])

        if self.poses:
            frames["left"].append_pose(
                Pose(torch.Tensor([float(x) for x in self.poses[idx].split(" ")] + [0, 0, 0, 1]).reshape(4, 4))
            )

        if self.folders["right"] is not None:
            frames["right"] = Frame(os.path.join(self.folders["right"], f"{idx:06d}.png"))
            frames["right"].baseline = 0.54
            frames["right"].append_cam_intrinsic(self.right_intrinsic)

        return frames
