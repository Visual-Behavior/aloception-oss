import os
import cv2
import torch
import numpy as np
from typing import Dict, List, Tuple

from alodataset import BaseDataset, Split, SplitMixin
from aloscene import Frame, Disparity, Mask, Flow, SceneFlow
from aloscene.camera_calib import CameraIntrinsic, CameraExtrinsic

from alodataset.utils.kitti import load_calib_cam_to_cam


class KittiStereoFlowSFlow2015(BaseDataset, SplitMixin):
    SPLIT_FOLDERS = {Split.TRAIN: "training", Split.TEST: "testing"}

    def __init__(
        self,
        name="kitti_stereo2015",
        sequence_start=10,
        sequence_end=11,
        grayscale=False,
        load: list = ["right", "disp_noc", "disp_occ", "flow_occ", "flow_noc", "scene_flow", "obj_map"],
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
        flow = (img - 2 ** 15) / 64.0
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

        # Compute depth from disparity at time T+1
        disparity = next_frame.disparity["warped_disp_noc"]
        next_depth = disparity.as_depth(baseline=0.54, camera_intrinsic=next_frame.cam_intrinsic)
        next_depth.occlusion = disparity.mask.clone()
        next_depth.append_cam_extrinsic(next_frame.cam_extrinsic)

        # Compute the points cloud from the depth at time T
        start_points = depth.as_points3d(camera_intrinsic=frame.cam_intrinsic).cpu().numpy()
        mask = ~np.isfinite(start_points)
        np.nan_to_num(start_points, copy=False, nan=-0.1, posinf=-0.1, neginf=-0.1)

        # Compute the points cloud from the depth at time T+1
        flow = next_frame.flow["flow_noc"][0].numpy()
        # Move the points by the the flow because the depth is warped
        y_points, x_points = torch.meshgrid(torch.arange(size[0]), torch.arange(size[1]))
        x_points = x_points + flow[0, ...]
        y_points = y_points + flow[1, ...]
        x_points.unsqueeze_(0).unsqueeze_(0)
        y_points.unsqueeze_(0).unsqueeze_(0)

        end_points = (
            next_depth.as_points3d(camera_intrinsic=next_frame.cam_intrinsic, points=(x_points, y_points))
            .cpu()
            .numpy()
        )
        mask = mask | ~np.isfinite(end_points)
        np.nan_to_num(end_points, copy=False, nan=-0.1, posinf=-0.1, neginf=-0.1)

        # Compute the scene flow from the points cloud
        mask = mask.any(axis=2)
        # Subtract removes unnecessary temporal dimension
        scene_flow = end_points - start_points
        scene_flow = scene_flow.transpose(0, 2, 1).reshape((3, size[0], size[1]))

        # Not sure if it is really usefull but here for security
        start_occlusion = depth.occlusion.cpu().numpy().astype(bool)
        # Need to squeze to remove the time dimension
        end_occlusion = np.squeeze(next_depth.occlusion.cpu().numpy().astype(bool), axis=0)

        # Fusion the occlusion mask
        mask2 = end_occlusion | start_occlusion
        mask = mask.reshape((1, size[0], size[1]))
        mask = mask2 | mask
        mask = Mask(mask, names=("C", "H", "W"))

        return SceneFlow(scene_flow, occlusion=mask, names=("C", "H", "W"))

    def getitem(self, idx) -> Dict[str, Frame]:
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
        if self.sample:
            return BaseDataset.__getitem__(self, idx)

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

    def _load_calib(self, path, idx):
        data = load_calib_cam_to_cam(
            os.path.join(path, "calib_cam_to_cam", f"{idx:06d}.txt"),
            os.path.join(path, "calib_velo_to_cam", f"{idx:06d}.txt"),
        )

        # Return only the parameters we are interested in.
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

    dataset = KittiStereoFlowSFlow2015(sequence_start=10, sequence_end=11, grayscale=False, sample=True)
    obj = dataset.getitem(randint(0, len(dataset) - 1))
    obj["left"].get_view().render()
