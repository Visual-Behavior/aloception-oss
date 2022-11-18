import os
import numpy as np
import torch
from typing import Dict

from alodataset import BaseDataset, Split, SplitMixin
from aloscene import Frame, BoundingBoxes2D, Labels, BoundingBoxes3D
from aloscene.camera_calib import CameraIntrinsic, CameraExtrinsic

from alodataset.utils.kitti import load_calib_cam_to_cam


class KittiObjectDataset(BaseDataset, SplitMixin):
    """
    Object Task from KITTI dataset.
    Parameters
    ----------
    name : str
        Name of the dataset.
    right_frame : bool
        If True, load the right frame.
    context_images : int
        Number of image before the main frame to load (max 3).
    split : Split
        Split of the dataset. Can be `Split.TRAIN` or `Split.TEST`.

    Examples
    --------
    >>> # Get all the training samples:
    >>> dataset = KittiObjectDataset(right_frame=True, context_images=3, split=Split.TRAIN)
    >>> # Get the annotated image from the left camera from the testing set:
    >>> dataset = KittiObjectDataset(right_frame=False, context_images=0, split=Split.TEST)
    """
    SPLIT_FOLDERS = {Split.TRAIN: "training", Split.TEST: "testing"}
    LABELS = ["Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram", "Misc", "DontCare"]

    def __init__(self, name="kitti_object", right_frame=False, context_images=3, **kwargs):
        super().__init__(name=name, **kwargs)

        assert context_images >= 0 and context_images <= 3, "You can only get 3 frames before the main frame"

        if self.sample:
            raise NotImplementedError("Sample is not implemented for KittiObjectDataset")

        self.split_folder = os.path.join(self.dataset_dir, self.get_split_folder())
        left_img_folder = os.path.join(self.split_folder, "image_2")
        n_samples = len(os.listdir(left_img_folder))

        self.items = {}
        for idx in range(n_samples):
            self.items[idx] = {
                "left": os.path.join(self.split_folder, f"image_2/{idx:06d}.png"),
                "right": os.path.join(self.split_folder, f"image_3/{idx:06d}.png") if right_frame else None,
                "label": os.path.join(self.split_folder, f"label_2/{idx:06d}.txt")
                if self.split == Split.TRAIN
                else None,
                "calib": os.path.join(self.split_folder, f"calib/{idx:06d}.txt"),
                "left_context_1": os.path.join(self.split_folder, f"prev_2/{idx:06d}_01.png")
                if context_images >= 1
                else None,
                "left_context_2": os.path.join(self.split_folder, f"prev_2/{idx:06d}_02.png")
                if context_images >= 2
                else None,
                "left_context_3": os.path.join(self.split_folder, f"prev_2/{idx:06d}_03.png")
                if context_images >= 3
                else None,
                "right_context_1": os.path.join(self.split_folder, f"prev_3/{idx:06d}_01.png")
                if context_images >= 1 and right_frame
                else None,
                "right_context_2": os.path.join(self.split_folder, f"prev_3/{idx:06d}_02.png")
                if context_images >= 2 and right_frame
                else None,
                "right_context_3": os.path.join(self.split_folder, f"prev_3/{idx:06d}_03.png")
                if context_images >= 3 and right_frame
                else None,
            }

    def getitem(self, idx) -> Dict[str, Frame]:

        if self.sample:
            return BaseDataset.__getitem__(self, idx)

        item = self.items[idx]
        calib = load_calib_cam_to_cam(item["calib"])
        frames = {}
        frames["left"] = Frame(item["left"])

        if item["right"] is not None:
            frames["right"] = Frame(item["right"])
            frames["right"].append_cam_intrinsic(CameraIntrinsic(np.c_[calib["K_cam3"], np.zeros(3)]))
            frames["right"].append_cam_extrinsic(CameraExtrinsic(calib["T_cam3_rect"]))
            frames["right"].baseline = calib["b_rgb"]
            frames["right"] = frames["right"].temporal()

        size = frames["left"].HW

        boxes2d = []
        boxes3d = []
        labels = []
        if item["label"] is not None:
            with open(item["label"], "r") as f:
                frame_info = f.readlines()

                for line in frame_info:
                    line = line.split()
                    x, y, w, h = float(line[4]), float(line[5]), float(line[6]), float(line[7])
                    boxes2d.append([x, y, w, h])
                    labels.append(self.LABELS.index(line[0]))
                    boxes3d.append(
                        [
                            float(line[11]),
                            # The center of the 3d box on Kitty is the center of the bottom face. We need to
                            # move it up by half the height of the box to correspond to the center of the box.
                            # Check kitti_tracking devkit for more info.
                            float(line[12]) - float(line[8]) / 2,
                            float(line[13]),
                            float(line[9]),
                            float(line[8]),
                            float(line[10]),
                            # The rotation of the 3d box on Kitty is based on the X axis. We need to rotate it
                            # to have same the rotation wanted by BoundingBoxes3D.
                            # Check kitti_object devkit for more info.
                            float(line[14]) + np.pi / 2,
                        ]
                    )

        labels = Labels(labels, labels_names=["boxes"])
        bounding_box = BoundingBoxes2D(boxes2d, boxes_format="xyxy", absolute=True, frame_size=size, labels=labels)
        boxe3d = BoundingBoxes3D(boxes3d)

        frames["left"].append_boxes2d(bounding_box)
        frames["left"].append_boxes3d(boxe3d)
        frames["left"].baseline = calib["b_rgb"]
        frames["left"].append_cam_intrinsic(CameraIntrinsic(np.c_[calib["K_cam2"], np.zeros(3)]))
        frames["left"].append_cam_extrinsic(CameraExtrinsic(calib["T_cam2_rect"]))
        frames["left"] = frames["left"].temporal()

        context_frames = [
            "left_context_1",
            "left_context_2",
            "left_context_3",
            "right_context_1",
            "right_context_2",
            "right_context_3",
        ]
        for context_frame in context_frames:
            if item[context_frame] is not None:
                frames[context_frame] = Frame(item[context_frame])
                side = 2 if context_frame.startswith("left") else 3
                frames[context_frame].append_cam_intrinsic(CameraIntrinsic(np.c_[calib[f"K_cam{side}"], np.zeros(3)]))
                frames[context_frame].append_cam_extrinsic(CameraExtrinsic(calib[f"T_cam{side}_rect"]))
                frames[context_frame].baseline = calib["b_rgb"]
                frames[context_frame] = frames[context_frame].temporal()

        ordered_frames = sorted([frame for frame in frames])[::-1]
        left = [frames[name] for name in ordered_frames if name.startswith("left")]
        right = [frames[name] for name in ordered_frames if name.startswith("right")]

        result = {}
        result["left"] = torch.cat(left, dim=0)
        if right:
            result["right"] = torch.cat(right, dim=0)

        return result


if __name__ == "__main__":
    from random import randint

    dataset = KittiObjectDataset(right_frame=True, context_images=2)
    obj = dataset.getitem(randint(0, len(dataset)))
    obj["left"].get_view().render()
