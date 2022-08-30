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
        grayscale=False,
        load: list = [
            "right",
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
        self.grayscale = grayscale
        self.load = load

        if self.sample:
            return

        assert sequence_start <= sequence_end, "sequence_start should be less than sequence_end"

        if "disp_noc" in load or "disp_occ" in load:
            assert sequence_start <= 11 and sequence_end >= 10, "Disparity is not available for this frame range"
            assert grayscale is False, "Disparity is only available in RGB"
        if "flow_occ" in load or "flow_noc" in load:
            assert sequence_start <= 10 and sequence_end >= 10, "Flow is not available for this frame range"
            assert grayscale is False, "Flow is only available in RGB"
        if "scene_flow" in load:
            assert "disp_noc" in load, "Scene flow need 'disp_noc'"
            assert sequence_start <= 10 and sequence_end >= 11, "Scene flow is not available for this frame range"
            assert grayscale is False, "Scene flow is only available in RGB"

        # Load sequence length
        self.split_folder = os.path.join(self.dataset_dir, self.get_split_folder())
        left_img_folder = os.path.join(self.split_folder, "image_2")
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

    def load_obj_map(self, path: str) -> List[Tuple[int, Mask]]:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        result = []

        # For each unique pixel value correspond a vehicule (execpt for 0)
        for px_value in np.unique(img)[1:]:
            result.append((px_value, Mask(np.array([img != px_value]), names=("C", "H", "W"))))
        return result

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
        disparity = frame.disparity["warped_disp_noc"]
        depth = disparity.as_depth(baseline=0.54, camera_intrinsic=frame.cam_intrinsic)
        depth.occlusion = disparity.mask.clone()
        depth.append_cam_extrinsic(frame.cam_extrinsic)
        # frame.append_depth(depth)

        # Compute depth from disparity at time T+1
        disparity = next_frame.disparity["warped_disp_noc"]
        next_depth = disparity.as_depth(baseline=0.54, camera_intrinsic=next_frame.cam_intrinsic)
        next_depth.occlusion = disparity.mask.clone()
        next_depth.append_cam_extrinsic(next_frame.cam_extrinsic)
        # next_frame.append_depth(next_depth)

        # Compute the points cloud from the depth at time T
        start_points = depth.as_points3d(camera_intrinsic=frame.cam_intrinsic).cpu().numpy()
        mask = ~np.isfinite(start_points)
        np.nan_to_num(start_points, copy=False, nan=-0.1, posinf=-0.1, neginf=-0.1)

        # Compute the points cloud from the depth at time T+1
        end_points = next_depth.as_points3d(camera_intrinsic=next_frame.cam_intrinsic).cpu().numpy()
        mask = mask | ~np.isfinite(end_points)
        np.nan_to_num(end_points, copy=False, nan=-0.1, posinf=-0.1, neginf=-0.1)

        # Compute the scene flow from the points cloud
        mask = mask.any(axis=2)
        scene_flow = end_points - start_points
        scene_flow = scene_flow.transpose(0, 2, 1).reshape((1, 3, size[0], size[1]))

        # Not sure if it is really usefull but here for security
        start_occlusion = depth.occlusion.cpu().numpy().astype(bool)
        end_occlusion = next_depth.occlusion.cpu().numpy().astype(bool)

        # Fusion the occlusion mask
        mask2 = end_occlusion | start_occlusion
        mask = mask.reshape((1, size[0], size[1]))
        mask = mask2 | mask
        mask = Mask(mask, names=("T", "C", "H", "W"))

        return SceneFlow(scene_flow, occlusion=mask, names=("T", "C", "H", "W"))

    def getitem(self, idx):
        """
        Function for compatibility
        Please check at getsequence
        """
        if self.sample:
            return BaseDataset.__getitem__(self, idx)
        return self.getsequence(idx)

    def getsequence(self, idx) -> Dict[int, Dict[str, Frame]]:
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
        calib = self._load_calib(self.split_folder, idx)

        # Load all the object before loading to create the correct number of dummy masks.
        obj_map = (
            self.load_obj_map(os.path.join(self.split_folder, f"obj_map/{idx:06d}_10.png"))
            if "obj_map" in self.load
            else None
        )

        # We need to load the sequance from the last to the first frame because we need information from the previous
        # frame to compute the scene flow.
        for index in range(self.sequence_end, self.sequence_start - 1, -1):
            sequence[index] = {}
            sequence[index]["left"] = Frame(os.path.join(self.split_folder, f"image_2/{idx:06d}_{index:02d}.png"))
            sequence[index]["left"].baseline = calib["baseline"]
            sequence[index]["left"].append_cam_intrinsic(calib["left_intrinsic"])
            sequence[index]["left"].append_cam_extrinsic(calib["left_extrinsic"])
            if "right" in self.load:
                sequence[index]["right"] = Frame(os.path.join(self.split_folder, f"image_3/{idx:06d}_{index:02d}.png"))
                sequence[index]["right"].baseline = ["baseline"]
                sequence[index]["right"].append_cam_intrinsic(calib["right_intrinsic"])
                sequence[index]["right"].append_cam_extrinsic(calib["right_extrinsic"])

            # Frames at index 10 and 11 are the only one who have ground truth in dataset.
            if index == 11:
                dummy_size = (2, sequence[index]["left"].H, sequence[index]["left"].W)
                dummy_disp_size = (1, sequence[index]["left"].H, sequence[index]["left"].W)
                if "disp_noc" in self.load:
                    sequence[11]["left"].append_disparity(
                        self.load_disp(os.path.join(self.split_folder, f"disp_noc_1/{idx:06d}_10.png"), "left"),
                        "warped_disp_noc",
                    )
                if "disp_occ" in self.load:
                    sequence[11]["left"].append_disparity(
                        self.load_disp(os.path.join(self.split_folder, f"disp_occ_1/{idx:06d}_10.png"), "left"),
                        "warped_disp_occ",
                    )
                if "flow_occ" in self.load:
                    sequence[index]["left"].append_flow(Flow.dummy(dummy_size, ("C", "H", "W")), "flow_occ")
                if "flow_noc" in self.load:
                    sequence[index]["left"].append_flow(Flow.dummy(dummy_size, ("C", "H", "W")), "flow_noc")
                if "scene_flow" in self.load:
                    scene_flow_size = (3, sequence[index]["left"].H, sequence[index]["left"].W)
                    sequence[index]["left"].append_scene_flow(SceneFlow.dummy(scene_flow_size, ("C", "H", "W")))
                if obj_map is not None:
                    for vehicule in obj_map:
                        sequence[index]["left"].append_mask(
                            Mask(torch.ones(dummy_disp_size), names=("C", "H", "W")), str(vehicule[0])
                        )
            elif index == 10:
                if "disp_noc" in self.load:
                    sequence[10]["left"].append_disparity(
                        self.load_disp(os.path.join(self.split_folder, f"disp_noc_0/{idx:06d}_10.png"), "left"),
                        "warped_disp_noc",
                    )
                if "disp_occ" in self.load:
                    sequence[10]["left"].append_disparity(
                        self.load_disp(os.path.join(self.split_folder, f"disp_occ_0/{idx:06d}_10.png"), "left"),
                        "warped_disp_occ",
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
                    sequence[10]["left"].append_scene_flow(scene_flow)
                if obj_map is not None:
                    for vehicule in obj_map:
                        sequence[10]["left"].append_mask(vehicule[1], str(vehicule[0]))
            else:
                dummy_size = (2, sequence[index]["left"].H, sequence[index]["left"].W)
                dummy_disp_size = (1, sequence[index]["left"].H, sequence[index]["left"].W)
                if "disp_noc" in self.load:
                    dummy_disp = Disparity.dummy(dummy_disp_size, ("C", "H", "W")).signed("left")
                    sequence[index]["left"].append_disparity(dummy_disp, "warped_disp_noc")
                if "disp_occ" in self.load:
                    dummy_disp = Disparity.dummy(dummy_disp_size, ("C", "H", "W")).signed("left")
                    sequence[index]["left"].append_disparity(dummy_disp, "warped_disp_occ")
                if "flow_occ" in self.load:
                    sequence[index]["left"].append_flow(Flow.dummy(dummy_size, ("C", "H", "W")), "flow_occ")
                if "flow_noc" in self.load:
                    sequence[index]["left"].append_flow(Flow.dummy(dummy_size, ("C", "H", "W")), "flow_noc")
                if "scene_flow" in self.load:
                    scene_flow_size = (3, sequence[index]["left"].H, sequence[index]["left"].W)
                    sequence[index]["left"].append_scene_flow(SceneFlow.dummy(scene_flow_size, ("C", "H", "W")))
                if obj_map is not None:
                    for vehicule in obj_map:
                        sequence[index]["left"].append_mask(
                            Mask(torch.ones(dummy_disp_size), names=("C", "H", "W")), str(vehicule[0])
                        )

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

    def transform_from_rot_trans(self, R, t):
        """Transforation matrix from rotation matrix and translation vector."""
        R = R.reshape(3, 3)
        t = t.reshape(3, 1)
        return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))

    def _load_calib_rigid(self, filepath):
        """Read a rigid transform calibration file as a numpy.array."""
        data = self.read_calib_file(filepath)
        return self.transform_from_rot_trans(data["R"], data["T"])

    def _load_calib_cam_to_cam(self, velo_to_cam_file, cam_to_cam_filepath):
        # We'll return the camera calibration as a dictionary
        data = {}

        # Load the rigid transformation from velodyne coordinates
        # to unrectified cam0 coordinates
        T_cam0unrect_velo = self._load_calib_rigid(velo_to_cam_file)
        data["T_cam0_velo_unrect"] = T_cam0unrect_velo

        # Load and parse the cam-to-cam calibration data
        filedata = self.read_calib_file(cam_to_cam_filepath)

        # Create 3x4 projection matrices
        P_rect_00 = np.reshape(filedata["P_rect_00"], (3, 4))
        P_rect_10 = np.reshape(filedata["P_rect_01"], (3, 4))
        P_rect_20 = np.reshape(filedata["P_rect_02"], (3, 4))
        P_rect_30 = np.reshape(filedata["P_rect_03"], (3, 4))

        data["P_rect_00"] = P_rect_00
        data["P_rect_10"] = P_rect_10
        data["P_rect_20"] = P_rect_20
        data["P_rect_30"] = P_rect_30

        # Create 4x4 matrices from the rectifying rotation matrices
        R_rect_00 = np.eye(4)
        R_rect_00[0:3, 0:3] = np.reshape(filedata["R_rect_00"], (3, 3))
        R_rect_10 = np.eye(4)
        R_rect_10[0:3, 0:3] = np.reshape(filedata["R_rect_01"], (3, 3))
        R_rect_20 = np.eye(4)
        R_rect_20[0:3, 0:3] = np.reshape(filedata["R_rect_02"], (3, 3))
        R_rect_30 = np.eye(4)
        R_rect_30[0:3, 0:3] = np.reshape(filedata["R_rect_03"], (3, 3))

        data["R_rect_00"] = R_rect_00
        data["R_rect_10"] = R_rect_10
        data["R_rect_20"] = R_rect_20
        data["R_rect_30"] = R_rect_30

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

        # Compute the velodyne to rectified camera coordinate transforms
        data["T_cam0_velo"] = T0.dot(R_rect_00.dot(T_cam0unrect_velo))
        data["T_cam1_velo"] = T1.dot(R_rect_00.dot(T_cam0unrect_velo))
        data["T_cam2_velo"] = T2.dot(R_rect_00.dot(T_cam0unrect_velo))
        data["T_cam3_velo"] = T3.dot(R_rect_00.dot(T_cam0unrect_velo))

        # Compute the camera intrinsics
        data["K_cam0"] = P_rect_00[0:3, 0:3]
        data["K_cam1"] = P_rect_10[0:3, 0:3]
        data["K_cam2"] = P_rect_20[0:3, 0:3]
        data["K_cam3"] = P_rect_30[0:3, 0:3]

        # Compute the stereo baselines in meters by projecting the origin of
        # each camera frame into the velodyne frame and computing the distances
        # between them
        p_cam = np.array([0, 0, 0, 1])
        p_velo0 = np.linalg.inv(data["T_cam0_velo"]).dot(p_cam)
        p_velo1 = np.linalg.inv(data["T_cam1_velo"]).dot(p_cam)
        p_velo2 = np.linalg.inv(data["T_cam2_velo"]).dot(p_cam)
        p_velo3 = np.linalg.inv(data["T_cam3_velo"]).dot(p_cam)

        data["b_gray"] = np.linalg.norm(p_velo1 - p_velo0)  # gray baseline
        data["b_rgb"] = np.linalg.norm(p_velo3 - p_velo2)  # rgb baseline

        return data

    def _load_calib(self, path, idx):
        """Load and compute intrinsic and extrinsic calibration parameters."""
        # We'll build the calibration parameters as a dictionary, then
        # convert it to a namedtuple to prevent it from being modified later
        data = {}

        # Load the rigid transformation from IMU to velodyne
        data["T_velo_imu"] = self._load_calib_rigid(os.path.join(path, "calib_imu_to_velo", f"{idx:06d}.txt"))

        # Load the camera intrinsics and extrinsics
        data.update(
            self._load_calib_cam_to_cam(
                os.path.join(path, "calib_velo_to_cam", f"{idx:06d}.txt"),
                os.path.join(path, "calib_cam_to_cam", f"{idx:06d}.txt"),
            )
        )

        # Return the parameters we want only
        result = {
            "baseline": data["b_gray"] if self.grayscale else data["b_rgb"],
            "left_intrinsic": CameraIntrinsic(np.c_[data["K_cam0"] if self.grayscale else data["K_cam2"], [0, 0, 0]]),
            "right_intrinsic": CameraIntrinsic(np.c_[data["K_cam1"] if self.grayscale else data["K_cam3"], [0, 0, 0]]),
            "left_extrinsic": CameraExtrinsic(data["T_cam0_rect"] if self.grayscale else data["T_cam2_rect"]),
            "right_extrinsic": CameraExtrinsic(data["T_cam1_rect"] if self.grayscale else data["T_cam3_rect"]),
        }
        return result


if __name__ == "__main__":
    from random import randint

    dataset = KittiStereoFlowSFlow2015(sequence_start=8, sequence_end=12, grayscale=False)
    obj = dataset.getitem(1)
    # print(obj)
    obj["right"].get_view().render()
