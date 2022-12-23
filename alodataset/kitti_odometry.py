import os
import torch
import numpy as np
from typing import List, Dict, Any

from alodataset import BaseDataset, SplitMixin, Split
from aloscene import Frame, Pose, CameraIntrinsic, CameraExtrinsic

from alodataset.utils.kitti import load_calib_cam_to_cam, sequence_indices


class KittiOdometryDataset(BaseDataset, SplitMixin):
    def __init__(
        self,
        name="kitti_odometry",
        sequences=None,
        grayscale=True,
        right_frame=True,
        sequence_size=3,
        skip=1,
        sequence_skip=1,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.grayscale = grayscale
        self.right_frame = right_frame
        self.sequence_size = sequence_size
        self.skip = skip
        self.sequence_skip = sequence_skip

        if self.sample:
            return

        self.dataset_dir = os.path.join(self.dataset_dir, "dataset")

        self.sequences = self._load_sequences(sequences)

        self.items: Dict[Any, Any] = {}

        self.seq_params = {}

        for seq in self.sequences:

            # Exploit data from calib.txt
            calib = load_calib_cam_to_cam(os.path.join(self.dataset_dir, "sequences", seq, "calib.txt"))

            self.seq_params[seq] = {}

            # Register sequence parameters
            self.seq_params[seq]["baseline"] = calib["b_gray" if self.grayscale else "b_rgb"]
            self.seq_params[seq]["left_intrinsic"] = CameraIntrinsic(
                np.c_[calib[f"K_cam{0 if grayscale else 2}"], np.zeros(3)]
            )
            self.seq_params[seq]["left_extrinsic"] = CameraExtrinsic(calib[f"T_cam{0 if grayscale else 2}_rect"])
            if right_frame:
                self.seq_params[seq]["right_intrinsic"] = CameraIntrinsic(
                    np.c_[calib[f"K_cam{1 if grayscale else 3}"], np.zeros(3)]
                )
                self.seq_params[seq]["right_extrinsic"] = CameraExtrinsic(calib[f"T_cam{1 if grayscale else 3}_rect"])

            with open(os.path.join(self.dataset_dir, "sequences", seq, "times.txt"), "r") as f:
                times = [float(x) for x in f.readlines()]

            poses = None
            if int(seq) < 11:
                with open(os.path.join(self.dataset_dir, "poses", f"{seq}.txt"), "r") as f:
                    poses = f.readlines()

            # Compute all the items.
            sequence_size = len(times)

            # Compute sequence indices
            temporal_sequences = sequence_indices(sequence_size, self.sequence_size, self.skip, self.sequence_skip)
            self.items.update(
                {
                    len(self.items)
                    + idx: {
                        "sequence": seq,
                        "temporal_sequence": temporal_seq,
                        "times": [times[x] for x in temporal_seq],
                        "poses": [poses[x] for x in temporal_seq] if poses else None,
                    }
                    for idx, temporal_seq in enumerate(temporal_sequences)
                }
            )

    def _load_sequences(self, sequences) -> List[str]:
        loaded_sequences = []

        if sequences is None:
            for seq in sorted(os.listdir(os.path.join(self.dataset_dir, "sequences"))):
                if (int(seq) <= 10 and self.split == Split.TRAIN) or (int(seq) >= 11 and self.split == Split.VAL):
                    loaded_sequences.append(seq)
        elif isinstance(sequences, int):
            if not os.path.exists(os.path.join(self.dataset_dir, "sequences", f"{sequences:02d}")):
                raise ValueError(f"Sequence {sequences:02d} does not exist")
            if (sequences <= 10 and self.split == Split.VAL) or (sequences >= 11 and self.split == Split.TRAIN):
                raise ValueError(f"Sequence {sequences:02d} is not in the {self.split} split")
            loaded_sequences.append(f"{sequences:02d}")
        elif isinstance(sequences, str):
            if not os.path.exists(os.path.join(self.dataset_dir, "sequences", sequences)):
                raise ValueError(f"Sequence {sequences} does not exist")
            if (int(sequences) <= 10 and self.split == Split.VAL) or (
                int(sequences) >= 11 and self.split == Split.TRAIN
            ):
                raise ValueError(f"Sequence {sequences} is not in the {self.split} split")
            loaded_sequences.append(sequences)
        elif isinstance(sequences, list):
            for seq in sequences:
                if not os.path.exists(os.path.join(self.dataset_dir, "sequences", seq)):
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
            left_frame = Frame(
                os.path.join(
                    self.dataset_dir,
                    "sequences",
                    sequence,
                    f"image_{0 if self.grayscale else 2}",
                    f"{seq:06d}.png",
                )
            )
            left_frame.baseline = self.seq_params[sequence]["baseline"]
            left_frame.append_cam_intrinsic(self.seq_params[sequence]["left_intrinsic"])
            left_frame.append_cam_extrinsic(self.seq_params[sequence]["left_extrinsic"])

            if self.split == Split.TRAIN:
                pose = Pose(
                    torch.Tensor([float(x) for x in item["poses"][id].split(" ")] + [0, 0, 0, 1]).reshape(4, 4)
                )
                left_frame.append_pose(pose)

            left.append(left_frame.temporal())

            if self.right_frame:
                right_frame = Frame(
                    os.path.join(
                        self.dataset_dir,
                        "sequences",
                        sequence,
                        f"image_{1 if self.grayscale else 3}",
                        f"{seq:06d}.png",
                    )
                )
                right_frame.baseline = self.seq_params[sequence]["baseline"]
                right_frame.append_cam_intrinsic(self.seq_params[sequence]["right_intrinsic"])
                right_frame.append_cam_extrinsic(self.seq_params[sequence]["right_extrinsic"])

                right.append(left_frame.temporal())

        frames = {}
        frames["left"] = torch.cat(left, dim=0)

        # Timestamps need to be added at the end because torch.cat can't merge them.
        frames["left"].timestamp = item["times"]

        if self.right_frame:
            frames["right"] = torch.cat(right, dim=0)
            frames["right"].timestamp = item["times"]

        return frames


if __name__ == "__main__":
    from random import randint

    dataset = KittiOdometryDataset(sequences=["00", "01"], sequence_skip=40, skip=28, sequence_size=5, sample=True)
    obj = dataset.getitem(randint(0, len(dataset) - 1))
    obj["left"].get_view().render()
