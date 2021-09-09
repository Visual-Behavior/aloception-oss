import os
import torch

from aloscene import Flow, Frame, Mask

from alodataset.sintel_base_dataset import SintelBaseDataset


class SintelFlowDataset(SintelBaseDataset):
    """
    MPI Sintel dataset for Optical Flow

    Parameters
    ----------

    cameras : list of string
        One frame is returned for each camera.
        Possible cameras : {'left'}.
    labels : list
        Labels that will be attached to the frame
        Possible labels : {'flow', 'flow_occ'}
    sequence_size: int
        Size of sequence to load
    sequence_skip: int
        Number of frames to skip between each element of the sequence
    **kwargs :
        aloscene.BaseDataset parameters
    """

    CAMERAS = ["left"]
    LABELS = ["flow", "flow_occ"]
    PASSES = ["albedo", "clean", "final"]

    def __init__(self, *args, **kwargs):
        super(SintelFlowDataset, self).__init__(name="SintelFlow", *args, **kwargs)

    def _left_img_dir(self, sintel_pass=None):
        """image directory for left camera"""
        sintel_pass = self.passes[0] if sintel_pass is None else sintel_pass
        return os.path.join(self.dataset_dir, "training", sintel_pass)

    def _add_sequence_element(self, el_id, sintel_pass, sintel_seq, features, labels, include_flow=True):
        def _folder(feature_or_label):
            return self._get_folder(sintel_seq, feature_or_label)

        # Features Images
        left_image = os.path.join(_folder(sintel_pass), f"frame_{el_id:04d}.png")
        if "left_image" in features:
            features["left_image"][-1].append(left_image)

        if include_flow:
            # Flow
            left_flow = os.path.join(_folder("flow"), f"frame_{el_id:04d}.flo")
            if "left_flow" in labels:
                labels["left_flow"][-1].append(left_flow)
            # Flow occlusion
            left_flow_occ = os.path.join(_folder("occlusions"), f"frame_{el_id:04d}.png")
            if "left_flow_occ" in labels:
                labels["left_flow_occ"][-1].append(left_flow_occ)

    @staticmethod
    def _load_flow(flow_label, occ_label, data, t):
        """
        Load a Flow object and append occlusion if it is found
        """
        if len(data[flow_label]) <= t:
            return None

        flow = Flow(data[flow_label][t])

        if occ_label in data:
            occ = Mask(data[occ_label][t])
            flow.append_occlusion(occ)

        return flow

    def _get_camera_frames(self, sequence_data, camera):
        """
        Load sequences frames for a specific camera
        """
        data = sequence_data[camera]
        frames = []
        for t in range(self.sequence_size):
            frame = Frame(data["image"][t]).temporal()

            if "flow" in data:
                flow = SintelFlowDataset._load_flow("flow", "flow_occ", data, t)
                if flow is not None:
                    frame.append_flow(flow, "flow_forward")

            frames.append(frame)

        frames = torch.cat(frames, dim=0).type(torch.float32)
        return frames


if __name__ == "__main__":
    dataset = SintelFlowDataset(sample=True)
    # show some frames at various indices
    for idx in [1, 2, 5]:
        frames = dataset.getitem(idx)["left"]
        frames.get_view().render()
