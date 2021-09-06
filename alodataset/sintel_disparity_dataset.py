import os
import torch

from aloscene import Disparity, Frame, Mask

from alodataset.sintel_base_dataset import SintelBaseDataset


class SintelDisparityDataset(SintelBaseDataset):
    """
    MPI Sintel dataset for Disparity

    Parameters
    ----------

    cameras : list of string
        One frame is returned for each camera.
        Possible cameras : {'left', 'right'}.
    labels : list
        Labels that will be attached to the frame
        Possible labels : {'disp', 'disp_occ'}
    sequence_size: int
        Size of sequence to load
    sequence_skip: int
        Number of frames to skip between each element of the sequence
    **kwargs :
        aloscene.BaseDataset parameters
    """

    CAMERAS = ["left", "right"]
    LABELS = ["disp", "disp_occ"]
    PASSES = ["clean", "final"]

    def __init__(self, *args, **kwargs):
        super(SintelDisparityDataset, self).__init__(name="SintelDisparity", *args, **kwargs)

    def _left_img_dir(self, sintel_pass=None):
        """some image dir to test sintel sequence"""
        sintel_pass = self.passes[0] if sintel_pass is None else sintel_pass
        pass_folder = f"{sintel_pass}_left"
        return os.path.join(self.dataset_dir, "training", pass_folder)

    def _add_sequence_element(self, el_id, sintel_pass, sintel_seq, features, labels, include_flow=True):
        def _folder(feature_or_label):
            return self._get_folder(sintel_seq, feature_or_label)

        # Features Images
        left_image = os.path.join(_folder(f"{sintel_pass}_left"), f"frame_{el_id:04d}.png")
        if "left_image" in features:
            features["left_image"][-1].append(left_image)
        right_image = os.path.join(_folder(f"{sintel_pass}_right"), f"frame_{el_id:04d}.png")
        if "right_image" in features:
            features["right_image"][-1].append(right_image)

        # Disparity
        left_disparity = os.path.join(_folder("disparities"), f"frame_{el_id:04d}.png")
        if "left_disp" in labels:
            labels["left_disp"][-1].append(left_disparity)
        # Disparity occlusion
        left_disp_occ = os.path.join(_folder("occlusions"), f"frame_{el_id:04d}.png")
        if "left_disp_occ" in labels:
            labels["left_disp_occ"][-1].append(left_disp_occ)

    @staticmethod
    def _load_disparity(disp_label, occ_label, data, t, camera):
        """
        Load a disparity map and append occlusion if it is found
        """
        if len(data[disp_label]) <= t:
            return None

        disp = Disparity(
            data[disp_label][t], names=("C", "H", "W"), disp_format="signed", camera_side=camera, png_negate=True
        ).temporal()

        if occ_label in data:
            occ = Mask(data[occ_label][t]).temporal()
            disp.append_occlusion(occ)

        return disp

    def _get_camera_frames(self, sequence_data, camera):
        """
        Load sequences frames for a specific camera
        """
        data = sequence_data[camera]
        frames = []
        for t in range(self.sequence_size):
            frame = Frame(data["image"][t]).temporal()

            if "flow" in data:
                flow = SintelDisparityDataset._load_flow("flow", "flow_occ", data, t)
                if flow is not None:
                    frame.append_flow(flow, "flow_forward")

            if "disp" in data:
                disp = SintelDisparityDataset._load_disparity("disp", "disp_occ", data, t, camera)
                if disp is not None:
                    frame.append_disparity(disp)

            frames.append(frame)

        frames = torch.cat(frames, dim=0).type(torch.float32)
        return frames


if __name__ == "__main__":
    dataset = SintelDisparityDataset(sequence_size=2)
    # show some frames at various indices
    for idx in [1, 15, 64]:
        frames = dataset.getitem(idx)["left"]
        frames.get_view().render()
