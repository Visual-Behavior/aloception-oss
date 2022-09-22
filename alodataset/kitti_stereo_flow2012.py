import os
import cv2
import torch
import numpy as np
from typing import Dict

from alodataset import BaseDataset, Split, SplitMixin
from aloscene import Frame, Disparity, Mask, Flow
from aloscene.camera_calib import CameraIntrinsic, CameraExtrinsic

from alodataset.utils.kitti import load_calib_cam_to_cam


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
            raise NotImplementedError("Sample mode is not implemented for KittiStereoFlow2012")

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

    def getitem(self, idx: int) -> Dict[str, Frame]:
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

        if self.sample:
            return BaseDataset.__getitem__(self, idx)

        # Read the calibration file
        calib = load_calib_cam_to_cam(os.path.join(self.split_folder, "calib", f"{idx:06d}.txt"))

        sequence: Dict[int, Dict[str, Frame]] = {}

        for index in range(self.sequence_end, self.sequence_start - 1, -1):
            sequence[index] = {}
            sequence[index]["left"] = Frame(
                os.path.join(self.split_folder, f"image_{0 if self.grayscale else 2}/{idx:06d}_{index:02d}.png")
            )
            sequence[index]["left"].baseline = 0.54
            sequence[index]["left"].append_cam_extrinsic(
                CameraExtrinsic(calib["T_cam0_rect"] if self.grayscale else calib["T_cam2_rect"])
            )
            sequence[index]["left"].append_cam_intrinsic(
                CameraIntrinsic(np.c_[calib["K_cam0"] if self.grayscale else calib["K_cam2"], [0, 0, 0]])
            )

            if "right" in self.load:
                sequence[index]["right"] = Frame(
                    os.path.join(self.split_folder, f"image_{1 if self.grayscale else 3}/{idx:06d}_{index:02d}.png")
                )
                sequence[index]["right"].baseline = 0.54
                sequence[index]["right"].append_cam_extrinsic(
                    CameraExtrinsic(calib["T_cam1_rect"] if self.grayscale else calib["T_cam3_rect"])
                )
                sequence[index]["right"].append_cam_intrinsic(
                    CameraIntrinsic(np.c_[calib["K_cam1"] if self.grayscale else calib["K_cam3"], [0, 0, 0]])
                )

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


if __name__ == "__main__":
    from random import randint

    dataset = KittiStereoFlow2012(grayscale=True, sequence_start=8)
    obj = dataset.getitem(randint(0, len(dataset)))
    obj["left"].get_view().render()
