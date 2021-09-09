import warnings

from alodataset import SintelFlowDataset, SintelDisparityDataset
from alodataset.sintel_base_dataset import SintelBaseDataset
from alodataset import BaseDataset, SequenceMixin


class SintelMultiDataset(BaseDataset, SequenceMixin):
    """Combination of MPI Sintel dataset for joint use of Optical Flow and Disparity

    Parameters
    ----------

    cameras : list of string
        One frame is returned for each camera.
        Possible cameras : {'left', 'right'}.
    labels : list
        Labels that will be attached to the frame
        Possible labels : {'flow', 'flow_occ'}
    passes : list
        Image passes {"albedo", "clean", "final"}

    **kwargs :
        aloscene.BaseDataset parameters
    """

    CAMERAS = ["left", "right"]
    FLOW_LABELS = ["flow", "flow_occ"]
    DISP_LABELS = ["disp", "disp_occ"]
    LABELS = FLOW_LABELS + DISP_LABELS
    PASSES = ["clean", "final"]
    SINTEL_SEQUENCES = SintelBaseDataset.SINTEL_SEQUENCES

    def __init__(self, cameras=None, labels=None, passes=None, sintel_sequences=None, *args, **kwargs):
        super(SintelMultiDataset, self).__init__(name="SintelMulti", *args, **kwargs)
        self.cameras = cameras if cameras is not None else self.CAMERAS
        self.labels = labels if labels is not None else self.LABELS
        self.passes = passes if passes is not None else self.PASSES
        self.sintel_sequences = sintel_sequences if sintel_sequences is not None else self.SINTEL_SEQUENCES

        self._init_flow_and_disp(*args, **kwargs)

    def _init_flow_and_disp(self, *args, **kwargs):

        self.disp_labels = set(self.DISP_LABELS).intersection(self.labels)
        self.flow_labels = set(self.FLOW_LABELS).intersection(self.labels)

        self.disp_required = len(self.disp_labels) > 0 or "right" in self.cameras
        self.flow_required = len(self.flow_labels) > 0

        if not (self.disp_required or self.flow_required):
            raise ValueError(
                "No disparity or flow labels required: use specific SintelFlowDataset or SintelDisparityDataset for images."
            )

        if "albedo" in self.passes and self.disp_required:
            raise ValueError("Albedo pass is not available in Sintel Disparity dataset")

        if "final" in self.passes and self.flow_required and self.disp_required:
            sintel_image_warning = (
                "[SINTEL WARNING] The images in final pass are slightly different for Flow and Disparity datasets"
                " ---> Images from Disparity dataset will be used. For more information, see http://sintel.is.tue.mpg.de/stereo"
            )
            warnings.warn(sintel_image_warning)

        if self.flow_required and "right" in self.cameras:
            sintel_flow_warning = "[SINTEL WARNING] Sintel Optical Flow dataset contains only left camera. Right frame will have no flow attached."
            warnings.warn(sintel_flow_warning)

        if self.flow_required and "left" not in self.cameras:
            raise ValueError("Left camera is mandatory if some flow labels are required.")

        self.flow_dataset = None
        self.disp_dataset = None

        if self.flow_required:
            flow_cameras = set(self.cameras).intersection(["left"])
            self.flow_dataset = SintelFlowDataset(
                flow_cameras, self.flow_labels, self.passes, self.sintel_sequences, *args, **kwargs
            )

        if self.disp_required:
            self.disp_dataset = SintelDisparityDataset(
                self.cameras, self.disp_labels, self.passes, self.sintel_sequences, *args, **kwargs
            )

    def getitem(self, idx):
        if self.flow_required and self.disp_required:
            frames_flow = self.flow_dataset.getitem(idx)
            frames_disp = self.disp_dataset.getitem(idx)
            for camera, frame in frames_flow.items():
                frames_disp[camera].flow = frame.flow
            return frames_disp

        elif self.flow_required:
            return self.flow_dataset.getitem(idx)

        elif self.disp_required:
            return self.disp_dataset.getitem(idx)

        else:
            raise Exception()

    def __len__(self):
        if self.disp_required:
            return len(self.disp_dataset)
        elif self.flow_required:
            return len(self.flow_required)
        else:
            raise Exception()

    def get_dataset_dir(self):
        return None

    def set_dataset_dir(self, dataset_dir: str):
        raise Exception("SintelMultiDataset has no dataset_dir")


if __name__ == "__main__":
    dataset = SintelMultiDataset(sample=True)
    # show some frames at various indices
    for idx in [1, 2, 5]:
        frames = dataset.getitem(idx)
        frames["left"].get_view().render()
        frames["right"].get_view().render()
