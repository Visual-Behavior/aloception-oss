import itertools
import torch
import os

from aloscene.io.disparity import load_disp_png
from aloscene import Frame, Flow, Mask, Disparity
from aloscene.utils.data_utils import DLtoLD
from alodataset import BaseDataset, SequenceMixin


class SintelBaseDataset(BaseDataset, SequenceMixin):
    """
    Abstract Base Class for MPI Sintel datasets

    Parameters
    ----------

    cameras : list of string
        One frame is returned for each camera.
        Possible cameras : {'left'}.
    labels : list
        Labels that will be attached to the frame
        Possible labels : {'disp', 'disp_occ', 'flow', 'flow_occ'}
    **kwargs :
        aloscene.BaseDataset parameters
    """

    CAMERAS = NotImplemented
    LABELS = NotImplemented
    PASSES = NotImplemented

    SINTEL_SEQUENCES = [
        "alley_1",
        "alley_2",
        "ambush_2",
        "ambush_4",
        "ambush_5",
        "ambush_6",
        "ambush_7",
        "bamboo_1",
        "bamboo_2",
        "bandage_1",
        "bandage_2",
        "cave_2",
        "cave_4",
        "market_2",
        "market_5",
        "market_6",
        "mountain_1",
        "shaman_2",
        "shaman_3",
        "sleeping_1",
        "sleeping_2",
        "temple_2",
        "temple_3",
    ]

    def __init__(self, cameras=None, labels=None, passes=None, sintel_sequences=None, name="Sintel_base", **kwargs):
        super(SintelBaseDataset, self).__init__(name=name, **kwargs)

        if self.sequence_skip != 0:
            raise NotImplementedError("Skipping frame is not yet implemented")

        self.cameras = cameras if cameras is not None else self.CAMERAS
        self.labels = labels if labels is not None else self.LABELS
        self.passes = passes if passes is not None else self.PASSES
        self.sintel_sequences = sintel_sequences if sintel_sequences is not None else self.SINTEL_SEQUENCES
        self._assert_inputs()

        self.items = self._get_sequences()

    def _assert_inputs(self):
        assert len(self.cameras) > 0
        assert len(self.passes) > 0
        assert len(self.sintel_sequences) > 0

    def _prepare_feature_labels(self):
        features = [f"{camera}_image" for camera in self.cameras]
        labels = [f"{camera}_{label}" for camera, label in itertools.product(self.cameras, self.labels)]
        features, labels = self._get_features_labels(features, labels)
        return features, labels

    def _get_sequences(self):
        features, labels = self._prepare_feature_labels()
        sequences = DLtoLD({**features, **labels})
        sequences = [self._split_cameras(sequence_data) for sequence_data in sequences]
        return sequences

    def _split_cameras(self, my_dict):
        splitted = {camera: {} for camera in self.cameras}
        for key, value in my_dict.items():
            camera, label = key.split("_", 1)
            splitted[camera][label] = value
        return splitted

    def _get_folder(self, sintel_seq, feature_or_label):
        dset_dir = self.dataset_dir
        return os.path.join(dset_dir, "training", feature_or_label, sintel_seq)

    @property
    def _left_img_dir(self, sintel_pass=None):
        """some image dir to test sintel sequence"""
        raise NotImplementedError()

    def _sintel_seq_len(self, sintel_seq):
        sequence_folder = os.path.join(self._left_img_dir(), sintel_seq)
        return len(os.listdir(sequence_folder))

    def _add_sequence_element(self, el_id, sintel_pass, sintel_seq, features, labels, include_flow):
        raise NotImplementedError()

    def _get_features_labels(self, selected_features, selected_labels):

        # Init features and labels list
        features = {}
        labels = {}
        for feature in selected_features:
            features[feature] = []
        for label in selected_labels:
            labels[label] = []

        # Iterate over all passes
        for sintel_pass in self.passes:
            for sintel_seq in self.sintel_sequences:

                # Create sequences
                for st_id in range(1, self._sintel_seq_len(sintel_seq) + 1 - self.sequence_size):

                    # Init sequence to empty lists
                    for feature in selected_features:
                        features[feature].append([])
                    for label in selected_labels:
                        labels[label].append([])

                    # Add elements to the sequence for each time step
                    for el_id in range(st_id, st_id + self.sequence_size):
                        include_flow = True if el_id + 1 < st_id + self.sequence_size else False
                        self._add_sequence_element(
                            el_id, sintel_pass, sintel_seq, features, labels, include_flow=include_flow
                        )

        return features, labels

    def _get_camera_frames(self, sequence_data, camera):
        """
        Load sequences frames for a specific camera
        """
        raise NotImplementedError()

    def _get_frames(self, sequence_data):
        frames = {}
        for camera in self.cameras:
            frames[camera] = self._get_camera_frames(sequence_data, camera)
        return frames

    def getitem(self, idx):
        sequence_data = self.items[idx]
        return self._get_frames(sequence_data)
