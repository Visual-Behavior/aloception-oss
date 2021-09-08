import itertools
import torch
import os

from aloscene.utils.data_utils import DLtoLD
from alodataset import BaseDataset, SequenceMixin, Split, SplitMixin

# from aloscene.io.disparity import load_disp

from aloscene import Frame, Flow, Mask, Disparity


class FlyingThings3DSubsetDataset(BaseDataset, SequenceMixin, SplitMixin):
    """
    FlyingThings3DSubsetDataset
    Synthetic data with flow and disparity

    Parameters
    ----------

    backward : bool
        if True, read dataset from t=0 to the end, then from the end to t=0.
    cameras : list of string
        One frame is returned for each camera.
        Possible cameras : {'left', 'right'}.
    labels : list
        Labels that will be attached to the frame
        Possible labels : {'disp', 'disp_occ', 'flow', 'flow_occ', 'flow_backward', 'flow_occ_backward'}
    **kwargs :
        aloscene.BaseDataset parameters
    """

    CAMERAS = ["left", "right"]
    LABELS = ["disp", "disp_occ", "flow", "flow_occ", "flow_backward", "flow_occ_backward"]

    def __init__(self, backward=False, cameras=None, labels=None, **kwargs):
        super(FlyingThings3DSubsetDataset, self).__init__(name="FlyingThings3DSubset", **kwargs)
        if self.sample:
            return
        if self.sequence_skip != 0 and any(
            i in labels for i in ["flow", "flow_occ", "flow_backward", "flow_occ_backward"]
        ):
            raise NotImplementedError(
                "Skipping frame is not yet implemented for FlyingThings3DSubsetDataset (because of Flow)"
            )
        self.images_format = "png"
        self.disparities_format = "pfm"
        self.train_val = "train" if self.split is Split.TRAIN else "val"
        self.backward = backward
        self.cameras = cameras if cameras is not None else self.CAMERAS
        self.labels = labels if labels is not None else self.LABELS
        self.items = self._get_sequences()

    def _get_sequences(self):
        features = [f"{camera}_image" for camera in self.cameras]
        labels = [f"{camera}_{label}" for camera, label in itertools.product(self.cameras, self.labels)]
        features, labels = self.get_features_labels(features, labels)
        sequences = DLtoLD({**features, **labels})
        sequences = [self._split_cameras(sequence_data) for sequence_data in sequences]
        return sequences

    def _split_cameras(self, my_dict):
        splitted = {camera: {} for camera in self.cameras}
        for key, value in my_dict.items():
            camera, label = key.split("_", 1)
            splitted[camera][label] = value
        return splitted

    @staticmethod
    def id_sf_path(dataset_dir, folder, ext, _id, train_val="train", left_right="left", future_past=None):
        # Get the required path
        if future_past is None:
            path = os.path.join(dataset_dir, "{}/{}/{}/{:07d}.{}".format(train_val, folder, left_right, _id, ext))
        else:
            path = os.path.join(
                dataset_dir, "{}/{}/{}/{}/{:07d}.{}".format(train_val, folder, left_right, future_past, _id, ext)
            )

        # Check if the past exist
        if os.path.exists(path):
            return [path]
        else:
            return False

    def get_featlabels_element(
        self, nb, selected_features, selected_labels, future_past="into_future", include_flow=True
    ):
        # Get an element of the sequence given the Number id
        features = {}
        labels = {}

        # Retrieve the images (path)
        features["left_image"] = (
            FlyingThings3DSubsetDataset.id_sf_path(
                self.dataset_dir, "image_clean", "png", nb, self.train_val, "left", None
            )
            if "left_image" in selected_features
            else ""
        )
        features["right_image"] = (
            FlyingThings3DSubsetDataset.id_sf_path(
                self.dataset_dir, "image_clean", "png", nb, self.train_val, "right", None
            )
            if "right_image" in selected_features
            else ""
        )

        # Retrieve the disparity (path)
        labels["left_disp"] = (
            FlyingThings3DSubsetDataset.id_sf_path(self.dataset_dir, "disparity", "pfm", nb, self.train_val, "left")
            if "left_disp" in selected_labels
            else ""
        )
        labels["right_disp"] = (
            FlyingThings3DSubsetDataset.id_sf_path(self.dataset_dir, "disparity", "pfm", nb, self.train_val, "right")
            if "right_disp" in selected_labels
            else ""
        )
        # Retrieve the occlusion disparity (path)
        labels["left_disp_occ"] = (
            FlyingThings3DSubsetDataset.id_sf_path(
                self.dataset_dir, "disparity_occlusions", "png", nb, self.train_val, "left"
            )
            if "left_disp_occ" in selected_labels
            else ""
        )
        labels["right_disp_occ"] = (
            FlyingThings3DSubsetDataset.id_sf_path(
                self.dataset_dir, "disparity_occlusions", "png", nb, self.train_val, "right"
            )
            if "right_disp_occ" in selected_labels
            else ""
        )

        if include_flow:  # The last element of each sequence should not include the flow
            # Retrieve the flow (path)
            labels["left_flow"] = (
                FlyingThings3DSubsetDataset.id_sf_path(
                    self.dataset_dir, "flow", "flo", nb, self.train_val, "left", future_past=future_past
                )
                if "left_flow" in selected_labels
                else ""
            )
            labels["right_flow"] = (
                FlyingThings3DSubsetDataset.id_sf_path(
                    self.dataset_dir, "flow", "flo", nb, self.train_val, "right", future_past=future_past
                )
                if "right_flow" in selected_labels
                else ""
            )
            labels["left_flow_backward"] = (
                FlyingThings3DSubsetDataset.id_sf_path(
                    self.dataset_dir, "flow", "flo", nb + 1, self.train_val, "left", future_past="into_past"
                )
                if "left_flow" in selected_labels
                else ""
            )
            labels["right_flow_backward"] = (
                FlyingThings3DSubsetDataset.id_sf_path(
                    self.dataset_dir, "flow", "flo", nb + 1, self.train_val, "right", future_past="into_past"
                )
                if "right_flow" in selected_labels
                else ""
            )

            # Retrieve the occlusion flow (path)
            labels["left_flow_occ"] = (
                FlyingThings3DSubsetDataset.id_sf_path(
                    self.dataset_dir, "flow_occlusions", "png", nb, self.train_val, "left", future_past=future_past
                )
                if "left_flow_occ" in selected_labels
                else ""
            )
            labels["right_flow_occ"] = (
                FlyingThings3DSubsetDataset.id_sf_path(
                    self.dataset_dir, "flow_occlusions", "png", nb, self.train_val, "right", future_past=future_past
                )
                if "right_flow_occ" in selected_labels
                else ""
            )
            labels["left_flow_occ_backward"] = (
                FlyingThings3DSubsetDataset.id_sf_path(
                    self.dataset_dir, "flow_occlusions", "png", nb + 1, self.train_val, "left", future_past="into_past"
                )
                if "left_flow_occ" in selected_labels
                else ""
            )
            labels["right_flow_occ_backward"] = (
                FlyingThings3DSubsetDataset.id_sf_path(
                    self.dataset_dir,
                    "flow_occlusions",
                    "png",
                    nb + 1,
                    self.train_val,
                    "right",
                    future_past="into_past",
                )
                if "right_flow_occ" in selected_labels
                else ""
            )

        # Manage valid frame data (string content isn't used, using fake generator)
        labels["left_disp_valid"] = ["left_disp_valid"] if "left_disp_valid" in selected_labels else ""
        labels["right_disp_valid"] = ["right_disp_valid"] if "right_disp_valid" in selected_labels else ""
        labels["left_disp_occ_valid"] = ["left_disp_occ_valid"] if "left_disp_occ_valid" in selected_labels else ""
        labels["right_disp_occ_valid"] = ["right_disp_occ_valid"] if "right_disp_occ_valid" in selected_labels else ""

        # Check if one of the required elements is False (not found)
        for key in features:
            if features[key] is False:
                return False, False
        for key in labels:
            if labels[key] is False:
                return False, False

        # Keep only the required features and labels
        features = {key: features[key] for key in features if features[key] != ""}
        labels = {key: labels[key] for key in labels if labels[key] != ""}

        return features, labels

    def _get_features_labels(self, selected_features, selected_labels, backward=False):
        """ """
        path = os.path.join(self.dataset_dir, "{}/image_clean/left".format(self.train_val))
        nb_image = len(os.listdir(path))
        # List of sequences
        features_dict = {}
        labels_dict = {}
        n_sequence = []

        # Go through the dataset (forward or backward)
        nb = 0 if backward is False else nb_image - 1
        while_cond = lambda x: x < nb_image if backward is False else x >= 0
        future_past = "into_future" if backward is False else "into_past"
        mark_nb = 0

        while while_cond(nb):

            # Get the features and the associated labels for this element number
            include_flow = True if len(n_sequence) + 1 < self.sequence_size else False
            n_features, n_labels = self.get_featlabels_element(
                nb, selected_features, selected_labels, future_past=future_past, include_flow=include_flow
            )

            # Add the features and the labels to the current sequence
            if n_features is not False and n_labels is not False and len(n_sequence) < self.sequence_size:
                mark_nb = nb
                n_sequence.append((n_features, n_labels))
            elif len(n_sequence) == self.sequence_size:
                features = {}
                labels = {}

                # Concat the features and the labels for this new sequence
                for n_features, n_labels in n_sequence:
                    features = {
                        key: features.get(key, []) + (n_features[key] if key in n_features else [])
                        for key in selected_features
                    }
                    labels = {
                        key: labels.get(key, []) + (n_labels[key] if key in n_labels else [])
                        for key in selected_labels
                    }

                # Append the features and the current sequences to the main dicts
                for key in selected_features:
                    features_dict[key] = [] if key not in features_dict else features_dict[key]
                    features_dict[key].append(features[key])
                for key in selected_labels:
                    labels_dict[key] = [] if key not in labels_dict else labels_dict[key]
                    labels_dict[key].append(labels[key])

                nb = mark_nb
                n_sequence = []
            else:
                n_sequence = []

            nb = nb + 1 + self.sequence_skip if backward is False else nb - 1 - self.sequence_skip

        return features_dict, labels_dict

    def get_features_labels(self, selected_features, selected_labels):
        forward_features, forward_labels = self._get_features_labels(
            selected_features, selected_labels, backward=False
        )
        # Merged the forward sequences with the backward sequences
        if self.backward is True:
            backward_features, backward_labels = self._get_features_labels(
                selected_features, selected_labels, backward=True
            )
            for key in forward_features:
                forward_features[key] += backward_features[key]
            for key in forward_labels:
                forward_labels[key] += backward_labels[key]
        return forward_features, forward_labels

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

    @staticmethod
    def _load_disparity(disp_label, occ_label, data, t, camera):
        """
        Load a disparity map and append occlusion if it is found
        """
        if len(data[disp_label]) <= t:
            return None

        disp = Disparity(data[disp_label][t], disp_format="signed", camera_side=camera).temporal()

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
                flow = FlyingThings3DSubsetDataset._load_flow("flow", "flow_occ", data, t)
                if flow is not None:
                    frame.append_flow(flow, "flow_forward")

            if "flow_backward" in data:
                flow_backward = FlyingThings3DSubsetDataset._load_flow("flow_backward", "flow_occ_backward", data, t)
                if flow_backward is not None:
                    frame.append_flow(flow_backward, "flow_backward")

            if "disp" in data:
                disp = FlyingThings3DSubsetDataset._load_disparity("disp", "disp_occ", data, t, camera)
                if disp is not None:
                    frame.append_disparity(disp)

            frames.append(frame)
        frames = torch.cat(frames, dim=0).type(torch.float32)
        return frames

    def get_frames(self, sequence_data):
        frames = {}
        for camera in self.cameras:
            frames[camera] = self._get_camera_frames(sequence_data, camera)
        return frames

    def getitem(self, idx):
        if self.sample:
            return BaseDataset.__getitem__(self, idx)
        sequence_data = self.items[idx]
        return self.get_frames(sequence_data)


if __name__ == "__main__":
    # import alodataset.transforms as T

    dataset = FlyingThings3DSubsetDataset(sample=True)
    for idx, frame in enumerate(dataset.stream_loader()):
        if idx > 3:
            break
        frame["left"].get_view().render()
