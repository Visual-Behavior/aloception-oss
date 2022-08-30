import os
import cv2
import torch
import numpy as np
from typing import Dict

from alodataset import BaseDataset, Split, SplitMixin
from aloscene import Frame, Disparity, Mask, Flow
from aloscene.camera_calib import CameraIntrinsic, CameraExtrinsic


class KittiStereoFlow2012(BaseDataset, SplitMixin):
    SPLIT_FOLDERS = {Split.TRAIN: "training", Split.TEST: "testing"}

    def __init__(
        self,
        name="kitti_stereo2012",
        grayscale=True,
        sequence_start=10,
        sequence_end=11,
        load: list = [
            "right",
            "disp_noc",
            "disp_occ",
            "disp_refl_noc",
            "disp_refl_occ",
            "flow_occ",
            "flow_noc",
        ],
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.grayscale = grayscale
        self.load = load
        self.sequence_start = sequence_start
        self.sequence_end = sequence_end

        if self.sample:
            return

        assert sequence_start <= sequence_end, "sequence_start should be less than sequence_end"

        if "disp_noc" in load or "disp_occ" in load or "disp_refl_noc" in load or "disp_refl_occ" in load:
            assert sequence_start <= 11 and sequence_end >= 10, "Disparity is not available for this frame range"
            assert grayscale is True, "Disparity is only available in grayscale"
        if "flow_occ" in load or "flow_noc" in load:
            assert sequence_start <= 10 and sequence_end >= 10, "Flow is not available for this frame range"
            assert grayscale is True, "Flow is only available in grayscale"

        self.split_folder = os.path.join(self.dataset_dir, self.get_split_folder())
        left_img_folder = os.path.join(self.split_folder, "image_0")
        self.len = len([f for f in os.listdir(left_img_folder) if f.endswith("_10.png")])

    def __len__(self):
        if self.sample:
            return len(self.items)
        return self.len

    def load_disp(self, disp_path: str, camera_side: str):
        img = cv2.imread(disp_path, cv2.IMREAD_UNCHANGED)
        disp = img / 256.0
        disp = torch.as_tensor(disp[None, ...], dtype=torch.float32)
        mask = disp <= 0
        mask = Mask(mask, names=("C", "H", "W"))
        disp = Disparity(disp, names=("C", "H", "W"), mask=mask, camera_side=camera_side).signed()
        return disp

    def load_flow(self, flow_path: str):
        img = cv2.imread(flow_path, cv2.IMREAD_UNCHANGED)
        mask = img[..., 0] == 0
        flow = (img - 2**15) / 64.0
        flow = torch.as_tensor(flow[..., 1:], dtype=torch.float32).permute(2, 0, 1)
        mask = mask.reshape((1, flow.shape[1], flow.shape[2]))
        mask = Mask(mask, names=("C", "H", "W"))
        flow = Flow(flow, names=("C", "H", "W"), mask=mask)
        return flow

    def load_disp_refl(self, disp_refl_path: str, camera_side: str):
        img = cv2.imread(disp_refl_path, cv2.IMREAD_UNCHANGED).astype(bool)
        disp_refl = torch.as_tensor(img[None, ...], dtype=torch.float32)
        mask = disp_refl <= 0
        mask = Mask(mask, names=("C", "H", "W"))
        disp_refl = Disparity(disp_refl, names=("C", "H", "W"), camera_side=camera_side, mask=mask).signed()
        return disp_refl

    def getitem(self, idx: int):
        """Function for compatibility

        Parameters
        ----------
        idx : int
            Index of the item

        Returns
        -------
        """
        if self.sample:
            return BaseDataset.__getitem__(self, idx)
        return self.getsequence(idx)

    def getsequence(self, idx: int) -> Dict[int, Dict[str, Frame]]:
        """
        Loads a sequence of frames from the dataset.

        Parameters
        ----------
        idx : int
            Index of the sequence.

        Returns
        -------
        Dict[int, Dict[str, Frame]]
            Dictionary of index beetween sequance_start and sequance_end.\n
            Each index is a dictionary of frames ("left" and maybe "right").
        """

        # Read the calibration file
        calib = self._load_calib(os.path.join(self.split_folder, "calib", f"{idx:06d}.txt"))

        sequence: Dict[int, Dict[str, Frame]] = {}

        for index in range(self.sequence_end, self.sequence_start - 1, -1):
            sequence[index] = {}
            sequence[index]["left"] = Frame(
                os.path.join(self.split_folder, f"image_{0 if self.grayscale else 2}/{idx:06d}_{index:02d}.png")
            )
            sequence[index]["left"].baseline = 0.54
            sequence[index]["left"].append_cam_extrinsic(calib["left_extrinsic"])
            sequence[index]["left"].append_cam_intrinsic(calib["left_intrinsic"])

            if "right" in self.load:
                sequence[index]["right"] = Frame(
                    os.path.join(self.split_folder, f"image_{1 if self.grayscale else 3}/{idx:06d}_{index:02d}.png")
                )
                sequence[index]["right"].baseline = 0.54
                sequence[index]["right"].append_cam_extrinsic(calib["right_extrinsic"])
                sequence[index]["right"].append_cam_intrinsic(calib["right_intrinsic"])

            # Frame at index 10 is the only one to have ground truth in dataset.
            if index == 10:
                if "disp_noc" in self.load:
                    sequence[10]["left"].append_disparity(
                        self.load_disp(os.path.join(self.split_folder, f"disp_noc/{idx:06d}_10.png"), "left"),
                        "disp_noc",
                    )
                if "disp_occ" in self.load:
                    sequence[10]["left"].append_disparity(
                        self.load_disp(os.path.join(self.split_folder, f"disp_occ/{idx:06d}_10.png"), "left"),
                        "disp_occ",
                    )
                if "disp_refl_noc" in self.load:
                    sequence[10]["left"].append_disparity(
                        self.load_disp_refl(
                            os.path.join(self.split_folder, f"disp_refl_noc/{idx:06d}_10.png"), "left"
                        ),
                        "disp_refl_noc",
                    )
                if "disp_refl_occ" in self.load:
                    sequence[10]["left"].append_disparity(
                        self.load_disp_refl(
                            os.path.join(self.split_folder, f"disp_refl_occ/{idx:06d}_10.png"), "left"
                        ),
                        "disp_refl_occ",
                    )
                if "flow_occ" in self.load:
                    sequence[10]["left"].append_flow(
                        self.load_flow(os.path.join(self.split_folder, f"flow_occ/{idx:06d}_10.png")), "flow_occ"
                    )
                if "flow_noc" in self.load:
                    sequence[10]["left"].append_flow(
                        self.load_flow(os.path.join(self.split_folder, f"flow_noc/{idx:06d}_10.png")), "flow_noc"
                    )
            else:
                dummy_size = (2, sequence[index]["left"].HW[0], sequence[index]["left"].HW[1])
                dummy_size_disp = (1, sequence[index]["left"].HW[0], sequence[index]["left"].HW[1])
                dummy_names = ("C", "H", "W")
                if "disp_noc" in self.load:
                    dummy_disp = Disparity.dummy(dummy_size_disp, dummy_names)
                    dummy_disp = dummy_disp.signed("left")
                    sequence[index]["left"].append_disparity(dummy_disp, "disp_noc")
                if "disp_occ" in self.load:
                    dummy_disp = Disparity.dummy(dummy_size_disp, dummy_names)
                    dummy_disp = dummy_disp.signed("left")
                    sequence[index]["left"].append_disparity(dummy_disp, "disp_occ")
                if "disp_refl_noc" in self.load:
                    dummy_disp = Disparity.dummy(dummy_size_disp, dummy_names)
                    dummy_disp = dummy_disp.signed("left")
                    sequence[index]["left"].append_disparity(dummy_disp, "disp_refl_noc")
                if "disp_refl_occ" in self.load:
                    dummy_disp = Disparity.dummy(dummy_size_disp, dummy_names)
                    dummy_disp = dummy_disp.signed("left")
                    sequence[index]["left"].append_disparity(dummy_disp, "disp_refl_occ")
                if "flow_occ" in self.load:
                    sequence[index]["left"].append_flow(Flow.dummy(dummy_size, dummy_names), "flow_occ")
                if "flow_noc" in self.load:
                    sequence[index]["left"].append_flow(Flow.dummy(dummy_size, dummy_names), "flow_noc")

            sequence[index]["left"] = sequence[index]["left"].temporal()
            if "right" in self.load:
                sequence[index]["right"] = sequence[index]["right"].temporal()

        result = {}

        left = [sequence[frame]["left"] for frame in range(self.sequence_start, self.sequence_end + 1)]
        result["left"] = torch.cat(left, dim=0)
        if "right" in self.load:
            right = [sequence[frame]["right"] for frame in range(self.sequence_start, self.sequence_end + 1)]
            result["right"] = torch.cat(right, dim=0)

        return result

    # https://github.com/utiasSTARS/pykitti/tree/master
    def read_calib_file(self, filepath):
        """Read in a calibration file and parse into a dictionary."""
        data = {}

        with open(filepath, "r") as f:
            for line in f.readlines():
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

        data["T0"] = T0
        data["T1"] = T1
        data["T2"] = T2
        data["T3"] = T3

        # Compute the camera intrinsics
        data["K_cam0"] = P_rect_00[0:3, 0:3]
        data["K_cam1"] = P_rect_10[0:3, 0:3]
        data["K_cam2"] = P_rect_20[0:3, 0:3]
        data["K_cam3"] = P_rect_30[0:3, 0:3]

        # Return only the parameters we care.
        result = {
            "left_intrinsic": CameraIntrinsic(np.c_[data["K_cam0"] if self.grayscale else data["K_cam2"], [0, 0, 0]]),
            "right_intrinsic": CameraIntrinsic(np.c_[data["K_cam1"] if self.grayscale else data["K_cam3"], [0, 0, 0]]),
            "left_extrinsic": CameraExtrinsic(data["T0"] if self.grayscale else data["T2"]),
            "right_extrinsic": CameraExtrinsic(data["T1"] if self.grayscale else data["T3"]),
        }
        return result


if __name__ == "__main__":
    from random import randint

    dataset = KittiStereoFlow2012(grayscale=True, sequence_start=8)
    obj = dataset.getitem(randint(0, len(dataset)))
    print(obj)
    obj["left"].get_view().render()
