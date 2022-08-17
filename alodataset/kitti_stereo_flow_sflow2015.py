import os
import cv2
import torch
import numpy as np
from typing import Dict, List, Tuple

from alodataset import BaseDataset, Split, SplitMixin
from aloscene import Frame, Disparity, Mask, Flow, SceneFlow
from aloscene.camera_calib import CameraIntrinsic, CameraExtrinsic


class KittiStereoFlowSFlow2015(BaseDataset, SplitMixin):
    SPLIT_FOLDERS = {Split.TRAIN: "training", Split.TEST: "testing"}

    def __init__(
        self,
        name="kitti_stereo2015",
        sequence_start=10,
        sequence_end=11,
        load: list = [
            "right_frame",
            "disp_noc",
            "disp_occ",
            "flow_occ",
            "flow_noc",
            "scene_flow",
            "obj_map",
        ],
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.sequence_start = sequence_start
        self.sequence_end = sequence_end
        self.load = load

        if self.sample:
            return

        assert sequence_start <= sequence_end, "sequence_start should be less than sequence_end"

        if "disp_noc" in load or "disp_occ" in load:
            assert sequence_start <= 11 and sequence_end >= 10, "Disparity is not available for this frame range"
        if "flow_occ" in load or "flow_noc" in load:
            assert sequence_start <= 10 and sequence_end >= 10, "Flow is not available for this frame range"
        if "scene_flow" in load:
            assert sequence_start <= 10 and sequence_end >= 11, "Scene flow is not available for this frame range"

        # Load sequence length
        self.split_folder = os.path.join(self.dataset_dir, self.get_split_folder())
        left_img_folder = os.path.join(self.split_folder, "image_2")
        self.len = len([f for f in os.listdir(left_img_folder) if f.endswith("_10.png")])

    def __len__(self):
        if self.sample:
            return len(self.items)
        return self.len

    def load_disp(self, disp_path: str):
        img = cv2.imread(disp_path, cv2.IMREAD_UNCHANGED)
        disp = img / 256.0
        disp = torch.as_tensor(disp[None, ...], dtype=torch.float32)
        valid_mask = disp > 0
        mask = Mask(valid_mask, names=("C", "H", "W"))
        disp = Disparity(disp, names=("C", "H", "W"), mask=mask)
        return disp

    def load_flow(self, flow_path: str):
        img = cv2.imread(flow_path, cv2.IMREAD_UNCHANGED)
        valid_mask = img[..., 0] == 1
        flow = (img - 2**15) / 64.0
        flow = torch.as_tensor(flow[..., 1:], dtype=torch.float32).permute(2, 0, 1)
        valid_mask = valid_mask.reshape((1, flow.shape[1], flow.shape[2]))
        mask = Mask(valid_mask, names=("C", "H", "W"))
        flow = Flow(flow, names=("C", "H", "W"), mask=mask)
        return flow

    def load_obj_map(self, path: str) -> List[Tuple[int, Mask]]:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        result = []

        # For each unique pixel value correspond a vehicule (execpt for 0)
        for px_value in np.unique(img)[1:]:
            result.append((px_value, Mask(np.array([img == px_value]), names=("C", "H", "W"))))

        return result

    @staticmethod
    def _get_cam_intrinsic(size):
        return CameraIntrinsic(focal_length=721.0, plane_size=size)

    @staticmethod
    def _fake_extrinsinc():
        x = torch.eye(4, dtype=torch.float32)
        return CameraExtrinsic(x)

    def extrinsic(self, idx: int) -> CameraExtrinsic:
        """Load extrinsic from corresponding file"""
        with open(os.path.join(self.split_folder, f"calib_cam_to_cam/{idx:06d}.txt"), "r") as f:
            file = f.readlines()
            rotation = [float(x) for x in file[5].split(" ")[1:]]
            translation = [float(x) for x in file[6].split(" ")[1:]]
            rotation = np.array(rotation).reshape(3, 3)
            translation = np.array(translation).reshape(3, 1)
            extrinsic = np.append(rotation, translation, axis=1)
            return CameraExtrinsic(np.append(extrinsic, np.array([[0, 0, 0, 1]]), axis=0))

    def scene_flow_from_disp(self, frame: Frame, next_frame: Frame) -> SceneFlow:
        """
        Compute scene flow from the disparity of 2 frames

        Parameters
        ----------
        frame : Frame
            Frame at time T.
        next_frame : Frame
            Frame at time T+1.

        Returns
        -------
        aloscene.SceneFlow
            Scene flow beetween the two frames.
        """

        size = frame.HW

        # Compute depth from disparity at time T
        disparity = frame.disparity["disp_noc"]
        depth = disparity.as_depth(baseline=0.54, camera_intrinsic=self._get_cam_intrinsic(size))
        depth.occlusion = disparity.mask.clone()
        depth.append_cam_extrinsic(self._fake_extrinsinc())
        frame.append_depth(depth)

        # Compute depth from disparity at time T+1
        disparity = next_frame.disparity["disp_noc"]
        next_depth = disparity.as_depth(baseline=0.54, camera_intrinsic=self._get_cam_intrinsic(size))
        next_depth.occlusion = disparity.mask.clone()
        next_depth.append_cam_extrinsic(self._fake_extrinsinc())
        next_frame.append_depth(next_depth)

        # Compute the points cloud from the depth at time T
        start_points = depth.as_points3d(camera_intrinsic=self._get_cam_intrinsic(size)).cpu().numpy()
        mask = ~np.isfinite(start_points)
        np.nan_to_num(start_points, copy=False, nan=-0.1, posinf=-0.1, neginf=-0.1)

        # Compute the points cloud from the depth at time T+1
        end_points = next_depth.as_points3d(camera_intrinsic=self._get_cam_intrinsic(size)).cpu().numpy()
        mask = mask | ~np.isfinite(end_points)
        np.nan_to_num(end_points, copy=False, nan=-0.1, posinf=-0.1, neginf=-0.1)

        # Compute the scene flow from the points cloud
        mask = mask.any(axis=1)
        scene_flow = end_points - start_points
        scene_flow = scene_flow.reshape((size[0], size[1], 3))

        # Not sure it is really usefull but here for security
        start_occlusion = depth.occlusion.cpu().numpy().astype(bool)
        end_occlusion = next_depth.occlusion.cpu().numpy().astype(bool)

        # Fusion the occlusion mask
        mask2 = end_occlusion | start_occlusion
        mask = mask.reshape((size[0], size[1]))
        mask2 = mask2.reshape((size[0], size[1]))
        mask = mask2 | mask
        mask = Mask(mask, names=("H", "W"))

        return SceneFlow(scene_flow, occlusion=mask)

    def getitem(self, idx):
        """
        Function for compatibility
        Please check at getsequence
        """
        if self.sample:
            return BaseDataset.__getitem__(self, idx)
        return self.getsequance(idx)

    def getsequance(self, idx) -> Dict[int, Dict[str, Frame]]:
        """
        Load a sequence of frames from the dataset.

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
        sequence: Dict[int, Dict[str, Frame]] = {}

        # We need to load the sequance from the last to the first frame because we need information from the previous
        # frame to compute the scene flow.
        for index in range(self.sequence_end, self.sequence_start - 1, -1):
            sequence[index] = {}
            sequence[index]["left"] = Frame(os.path.join(self.split_folder, f"image_2/{idx:06d}_{index:02d}.png"))
            size = sequence[index]["left"].HW
            sequence[index]["left"].baseline = 0.54
            sequence[index]["left"].append_cam_intrinsic(self._get_cam_intrinsic(size))
            sequence[index]["left"].append_cam_extrinsic(self.extrinsic(idx))
            if "right" in self.load:
                sequence[index]["right"] = Frame(os.path.join(self.split_folder, f"image_3/{idx:06d}_{index:02d}.png"))
                sequence[index]["right"].baseline = 0.54
                sequence[index]["right"].append_cam_intrinsic(self._get_cam_intrinsic(size))
                sequence[index]["right"].append_cam_extrinsic(self.extrinsic(idx))

            # Frames at index 10 and 11 are the only one who have ground truth in dataset.
            if index == 11:
                if "disp_noc" in self.load:
                    sequence[11]["left"].append_disparity(
                        self.load_disp(os.path.join(self.split_folder, f"disp_noc_1/{idx:06d}_10.png")), "disp_noc"
                    )
                if "disp_occ" in self.load:
                    sequence[11]["left"].append_disparity(
                        self.load_disp(os.path.join(self.split_folder, f"disp_occ_1/{idx:06d}_10.png")), "disp_occ"
                    )
            if index == 10:
                if "disp_noc" in self.load:
                    sequence[10]["left"].append_disparity(
                        self.load_disp(os.path.join(self.split_folder, f"disp_noc_0/{idx:06d}_10.png")), "disp_noc"
                    )
                if "disp_occ" in self.load:
                    sequence[10]["left"].append_disparity(
                        self.load_disp(os.path.join(self.split_folder, f"disp_occ_0/{idx:06d}_10.png")), "disp_occ"
                    )
                if "flow_occ" in self.load:
                    sequence[10]["left"].append_flow(
                        self.load_flow(os.path.join(self.split_folder, f"flow_occ/{idx:06d}_10.png")), "flow_occ"
                    )
                if "flow_noc" in self.load:
                    sequence[10]["left"].append_flow(
                        self.load_flow(os.path.join(self.split_folder, f"flow_noc/{idx:06d}_10.png")), "flow_noc"
                    )
                if "scene_flow" in self.load:
                    # The scene flow cannot be added currently because of a bug in his representation.
                    scene_flow = self.scene_flow_from_disp(sequence[10]["left"], sequence[11]["left"])
                    # sequence[10]["left"].append_scene_flow(scene_flow)
                if "obj_map" in self.load:
                    for vehicule in self.load_obj_map(
                        os.path.join(self.split_folder, f"obj_map/{idx:06d}_10.png")
                    ):
                        sequence[10]["left"].append_mask(vehicule[1], str(vehicule[0]))

        return sequence
