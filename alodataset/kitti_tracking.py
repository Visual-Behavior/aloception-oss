import os
import torch
import numpy as np
from typing import List, Dict, Any, Union
from collections import defaultdict

from alodataset import BaseDataset, SplitMixin, Split
from aloscene import Frame, CameraIntrinsic, CameraExtrinsic, BoundingBoxes2D, Labels, BoundingBoxes3D

from alodataset.utils.kitti import load_calib_cam_to_cam, sequence_indices


class KittiTrackingDataset(BaseDataset, SplitMixin):
    SPLIT_FOLDERS = {Split.TRAIN: "training", Split.VAL: "training", Split.TEST: "testing"}
    LABELS = ["Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram", "Misc", "DontCare", "Person"]

    def __init__(
        self,
        name="kitti_tracking",
        sequences: Union[int, List[int], None] = None,
        right_frame=True,
        sequence_size=3,
        skip=1,
        sequence_skip=1,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.right_frame = right_frame
        self.sequence_size = sequence_size
        self.skip = skip
        self.sequence_skip = sequence_skip

        if self.sample:
            raise NotImplementedError("Sample mode is not implemented for KittiTrackingDataset")

        self.dataset_dir = os.path.join(self.dataset_dir, self.get_split_folder())

        self.sequences = self._load_sequences(sequences)

        self.items: Dict[Any, Any] = {}

        self.seq_params = {}

        for seq in self.sequences:

            # Exploit data from calib.txt
            calib = self._load_calib(os.path.join(self.dataset_dir, "calib", f"{seq}.txt"))

            self.seq_params[seq] = {}

            # Register sequence parameters
            # self.seq_params[sequence]["baseline"] = calib"b_gra"_rgb"]
            self.seq_params[seq]["left_intrinsic"] = calib["left_intrinsic"]
            self.seq_params[seq]["left_extrinsic"] = calib["left_extrinsic"]
            self.seq_params[seq]["baseline"] = calib["baseline"]
            if right_frame:
                self.seq_params[seq]["right_intrinsic"] = calib["right_intrinsic"]
                self.seq_params[seq]["right_extrinsic"] = calib["right_extrinsic"]

            with open(os.path.join(self.dataset_dir, "label_02", f"{seq}.txt"), "r") as f:
                labels: defaultdict[int, List[List[str]]] = defaultdict(list)
                for line in f:
                    line = line.split()
                    labels[int(line[0])].append(line[1:])

            # Compute all the items.
            sequence_size = len(os.listdir(os.path.join(self.dataset_dir, "image_02", seq)))

            # Compute sequence indices
            temporal_sequences = sequence_indices(sequence_size, self.sequence_size, self.skip, self.sequence_skip)
            self.items.update(
                {
                    len(self.items)
                    + idx: {
                        "sequence": seq,
                        "temporal_sequence": temporal_seq,
                        "labels": [labels[x] for x in temporal_seq] if labels else None,
                    }
                    for idx, temporal_seq in enumerate(temporal_sequences)
                }
            )

    def _load_sequences(self, sequences) -> List[str]:
        if sequences is None:
            sequences = []
            for seq in sorted(os.listdir(os.path.join(self.dataset_dir, "image_02"))):
                if (int(seq) <= 10 and self.split == Split.TRAIN) or (int(seq) >= 11 and self.split == Split.VAL):
                    sequences.append(seq)
            return sequences

        if isinstance(sequences, int):
            sequences = [f"{sequences:04d}"]
        elif isinstance(sequences, str):
            sequences = [sequences]

        loaded_sequences = []
        for seq in sequences:
            seq = f"{seq:04d}" if isinstance(seq, int) else seq
            if not os.path.exists(os.path.join(self.dataset_dir, "image_02", seq)):
                raise ValueError(f"Sequence {seq} does not exist")
            if (int(seq) <= 10 and self.split == Split.VAL) or (int(seq) >= 11 and self.split == Split.TRAIN):
                raise ValueError(f"Sequence {seq} is not in the {self.split} split")
            loaded_sequences.append(seq)

        return loaded_sequences

    def getitem(self, idx: int) -> Dict[str, Frame]:
        """
        Loads a single frame from the dataset.

        Parameters
        ----------
        idx : int
            Index of the frame to load.

        Returns
        -------
        Dict[str, Frame]
            Dictionary with the loaded frame and pose.
        """

        if self.sample:
            return BaseDataset.__getitem__(self, idx)

        item = self.items[idx]

        left = []
        right = []
        sequence: str = item["sequence"]  # Number of the sequence

        for id, seq in enumerate(item["temporal_sequence"]):

            labels = []
            categories = []
            labels_3d = []
            categories_3d = []
            boxes2d = []
            boxes3d = []

            for box in item["labels"][id]:
                x, y, w, h = float(box[5]), float(box[6]), float(box[7]), float(box[8])
                boxes2d.append([x, y, w, h])
                labels.append(int(box[0]))  # track_id
                categories.append(self.LABELS.index(box[1]))  # type

                # If the object caterory is "Don't care", there is no 3D box.
                if box[1] != "DontCare":
                    boxes3d.append(
                        [
                            float(box[12]),
                            # The center of the 3d box on Kitty is the center of the bottom face. We need to
                            # move it up by half the height of the box to correspond to the center of the box.
                            # Check kitti_tracking devkit for more info.
                            float(box[13]) - float(box[9]) / 2,
                            float(box[14]),
                            float(box[10]),
                            float(box[9]),
                            float(box[11]),
                            # The rotation of the 3d box on Kitty is based on the X axis. We need to rotate it
                            # to have same the rotation wanted by BoundingBoxes3D.
                            # Check kitti_object devkit for more info.
                            float(box[15]) + np.pi / 2,
                        ]
                    )
                    labels_3d.append(int(box[0]))  # track_id
                    categories_3d.append(self.LABELS.index(box[1]))  # type

            left_frame = Frame(
                os.path.join(
                    self.dataset_dir,
                    "image_02",
                    sequence,
                    f"{seq:06d}.png",
                )
            )

            if boxes2d:
                labels = Labels(labels, labels_names=["boxes"])
                bounding_box = BoundingBoxes2D(boxes2d, boxes_format="xyxy", absolute=True, frame_size=left_frame.HW)
                bounding_box.append_labels(labels, "track_id")
                bounding_box.append_labels(Labels(categories, labels_names=self.LABELS), "categories")
                left_frame.append_boxes2d(bounding_box)
            if boxes3d:
                labels_3d = Labels(labels_3d, labels_names=["boxes"])
                boxe3d = BoundingBoxes3D(
                    boxes3d,
                    labels={"track_id": labels_3d, "categories": Labels(categories_3d, labels_names=self.LABELS)},
                )
                left_frame.append_boxes3d(boxe3d)

            left_frame.baseline = self.seq_params[sequence]["baseline"]
            left_frame.append_cam_extrinsic(CameraExtrinsic(self.seq_params[sequence]["left_extrinsic"]))
            left_frame.append_cam_intrinsic(CameraIntrinsic(self.seq_params[sequence]["left_intrinsic"]))

            # Need to create temporal dimension for future fusion.
            left.append(left_frame.temporal())

            if self.right_frame:
                right_frame = Frame(
                    os.path.join(
                        self.dataset_dir,
                        "image_03",
                        sequence,
                        f"{seq:06d}.png",
                    )
                )
                right_frame.append_cam_intrinsic(CameraIntrinsic(self.seq_params[sequence]["right_intrinsic"]))
                right_frame.append_cam_extrinsic(CameraExtrinsic(self.seq_params[sequence]["right_extrinsic"]))
                right_frame.baseline = self.seq_params[sequence]["baseline"]
                right.append(right_frame.temporal())

        frames = {}
        frames["left"] = torch.cat(left, dim=0)

        if self.right_frame:
            frames["right"] = torch.cat(right, dim=0)

        return frames

    def _load_calib(self, calib_filepath):
        data = load_calib_cam_to_cam(calib_filepath)

        # Return only the parameters we are interested in.
        result = {
            "left_intrinsic": np.c_[data["K_cam2"], [0, 0, 0]],
            "right_intrinsic": np.c_[data["K_cam3"], [0, 0, 0]],
            "left_extrinsic": data["T_cam2_rect"],
            "right_extrinsic": data["T_cam3_rect"],
            "baseline": data["b_rgb"],
        }
        return result


if __name__ == "__main__":
    from random import randint

    dataset = KittiTrackingDataset(right_frame=False)
    obj = dataset.getitem(randint(0, len(dataset)))
    obj["left"].get_view().render()
