import os
import numpy as np
import cv2

from alodataset import BaseDataset, Split, SplitMixin
from aloscene import Frame, Mask, Labels
from aloscene.camera_calib import CameraIntrinsic, CameraExtrinsic

LABELS = ["Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram", "Misc", "DontCare"]


class KittiRoadDataset(BaseDataset, SplitMixin):
    SPLIT_FOLDERS = {Split.TRAIN: "training", Split.TEST: "testing"}

    def __init__(self, name="kitti_road", right_frame=False, grayscale=False, environement="um", obj="road", **kwargs):
        super().__init__(name=name, **kwargs)

        assert environement in ["um", "umm", "uu"], "Environement must be in ['um', 'umm', 'uu']"
        assert obj == "road" or (obj == "lane" and environement == "um"), "Type must be 'road' or 'lane'"

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

    def __len__(self):
        return len(self.items)

    def getitem(self, idx):
        item = self.items[idx]
        calib = self._load_calib(item["calib"])

        frames = {}
        if item["right"] is not None:
            frames["right"] = Frame(item["right"])
            frames["right"].append_cam_extrinsic(CameraExtrinsic(calib[f"T_cam{1 if self.grayscale else 3}_rect"]))
            frames["right"].append_cam_intrinsic(CameraIntrinsic(calib[f"K_cam{1 if self.grayscale else 3}"]))
        frames["left"] = Frame(item["left"])
        frames["left"].append_cam_intrinsic(
            CameraIntrinsic(np.c_[calib[f"K_cam{0 if self.grayscale else 2}"], np.zeros(3)])
        )
        frames["left"].append_cam_extrinsic(CameraExtrinsic(calib[f"T_cam{0 if self.grayscale else 2}_rect"]))

        if item["ground_truth"] is not None:
            names = [self.obj, "valid_area", "invalid_area"]
            color_codes = [[0, 255], [2, 255], [2, 0]]
            # Cv2 loads images in BGR format
            ground_truth = cv2.imread(item["ground_truth"], cv2.IMREAD_UNCHANGED)
            print(ground_truth.shape, ground_truth)
            for name, color_code in zip(names, color_codes):
                area = ~np.array(ground_truth[:, :, color_code[0]] == color_code[1])
                print(area.shape, area)
                area = np.expand_dims(area, axis=0)
                area = Mask(area, names=("C", "H", "W"))
                frames["left"].append_mask(area, name=name)

        return frames

    # https://github.com/utiasSTARS/pykitti/tree/master
    def read_calib_file(self, filepath):
        """Read in a calibration file and parse into a dictionary."""
        data = {}

        with open(filepath, "r") as f:
            for line in f.readlines():
                if line == "\n":
                    continue
                key, value = line.split(":", 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass

        return data

    def _load_calib(self, calib_filepath):
        """Load and compute intrinsic and extrinsic calibration parameters."""
        # We'll build the calibration parameters as a dictionary, then
        # convert it to a namedtuple to prevent it from being modified later
        data = {}

        # Load the calibration file
        filedata = self.read_calib_file(calib_filepath)

        # Create 3x4 projection matrices
        P_rect_00 = np.reshape(filedata["P0"], (3, 4))
        P_rect_10 = np.reshape(filedata["P1"], (3, 4))
        P_rect_20 = np.reshape(filedata["P2"], (3, 4))
        P_rect_30 = np.reshape(filedata["P3"], (3, 4))

        data["P_rect_00"] = P_rect_00
        data["P_rect_10"] = P_rect_10
        data["P_rect_20"] = P_rect_20
        data["P_rect_30"] = P_rect_30

        # Compute the rectified extrinsics from cam0 to camN
        T0 = np.eye(4)
        T0[0, 3] = P_rect_00[0, 3] / P_rect_00[0, 0]
        T1 = np.eye(4)
        T1[0, 3] = P_rect_10[0, 3] / P_rect_10[0, 0]
        T2 = np.eye(4)
        T2[0, 3] = P_rect_20[0, 3] / P_rect_20[0, 0]
        T3 = np.eye(4)
        T3[0, 3] = P_rect_30[0, 3] / P_rect_30[0, 0]

        data["T_cam0_rect"] = T0
        data["T_cam1_rect"] = T1
        data["T_cam2_rect"] = T2
        data["T_cam3_rect"] = T3

        # Compute the camera intrinsics
        data["K_cam0"] = P_rect_00[0:3, 0:3]
        data["K_cam1"] = P_rect_10[0:3, 0:3]
        data["K_cam2"] = P_rect_20[0:3, 0:3]
        data["K_cam3"] = P_rect_30[0:3, 0:3]

        return data


if __name__ == "__main__":
    from random import randint

    dataset = KittiRoadDataset()
    obj = dataset.getitem(randint(0, len(dataset)))
    print("final", obj["left"].shape, obj)
    obj["left"].get_view().render()
