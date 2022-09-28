import os
import numpy as np
import cv2
from typing import Dict

from alodataset import BaseDataset, Split, SplitMixin
from aloscene import Frame, Mask
from aloscene.camera_calib import CameraIntrinsic, CameraExtrinsic

from alodataset.utils.kitti import load_calib_cam_to_cam


class KittiRoadDataset(BaseDataset, SplitMixin):
    SPLIT_FOLDERS = {Split.TRAIN: "training", Split.TEST: "testing"}

    def __init__(self, name="kitti_road", right_frame=False, grayscale=False, environement="um", obj="road", **kwargs):
        super().__init__(name=name, **kwargs)

        assert environement in ["um", "umm", "uu"], "Environement must be in ['um', 'umm', 'uu']"
        assert obj == "road" or (obj == "lane" and environement == "um"), "Type must be 'road' or 'lane'"

        if self.sample:
            raise NotImplementedError("Sample mode is not implemented for KittiRoadDataset")

        self.obj = obj
        self.grayscale = grayscale

        self.split_folder = os.path.join(self.dataset_dir, "data_road", self.get_split_folder())
        left_img_folder = os.path.join(self.split_folder, "image_2")
        n_samples = len([file for file in os.listdir(left_img_folder) if file.startswith(f"{environement}_")])
        self.extentions_dir = os.path.join(self.split_folder, self.get_split_folder())

        self.items = {}
        for idx in range(n_samples):
            self.items[idx] = {
                "left": os.path.join(self.extentions_dir, f"image_0/{environement}_{idx:06d}.png")
                if grayscale
                else os.path.join(self.split_folder, f"image_2/{environement}_{idx:06d}.png"),
                "right": os.path.join(self.extentions_dir, f"image_{1 if grayscale else 3}/{idx:06d}.png")
                if right_frame
                else None,
                "ground_truth": os.path.join(self.split_folder, f"gt_image_2/{environement}_{obj}_{idx:06d}.png")
                if self.split == Split.TRAIN and grayscale is False
                else None,
                "calib": os.path.join(self.split_folder, f"calib/{environement}_{idx:06d}.txt"),
            }

    def getitem(self, idx) -> Dict[str, Frame]:

        if self.sample:
            return BaseDataset.__getitem__(self, idx)
        item = self.items[idx]
        calib = load_calib_cam_to_cam(item["calib"])
        frames = {}

        if item["right"] is not None:
            frames["right"] = Frame(item["right"])
            frames["right"].append_cam_extrinsic(CameraExtrinsic(calib[f"T_cam{1 if self.grayscale else 3}_rect"]))
            frames["right"].append_cam_intrinsic(CameraIntrinsic(calib[f"K_cam{1 if self.grayscale else 3}"]))
            frames["right"].baseline = calib["b_gray" if self.grayscale else "b_rgb"]

        frames["left"] = Frame(item["left"])
        frames["left"].append_cam_intrinsic(
            CameraIntrinsic(np.c_[calib[f"K_cam{0 if self.grayscale else 2}"], np.zeros(3)])
        )
        frames["left"].append_cam_extrinsic(CameraExtrinsic(calib[f"T_cam{0 if self.grayscale else 2}_rect"]))
        frames["left"].baseline = calib["b_gray" if self.grayscale else "b_rgb"]

        if item["ground_truth"] is not None:
            names = [self.obj, "valid_area", "invalid_area"]
            color_codes = [[0, 255], [2, 255], [2, 0]]
            # Cv2 loads images in BGR format
            ground_truth = cv2.imread(item["ground_truth"], cv2.IMREAD_UNCHANGED)
            for name, color_code in zip(names, color_codes):
                area = ~np.array(ground_truth[:, :, color_code[0]] == color_code[1])
                area = np.expand_dims(area, axis=0)
                area = Mask(area, names=("C", "H", "W"))
                frames["left"].append_mask(area, name=name)

        return frames


if __name__ == "__main__":
    from random import randint

    dataset = KittiRoadDataset()
    obj = dataset.getitem(randint(0, len(dataset)))
    obj["left"].get_view().render()
