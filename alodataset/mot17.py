# from aloscene.tensors.spatial_augmented_tensor import SpatialAugmentedTensor
# from typing import Dict, List, Tuple
import more_itertools
import numpy as np
import torch
import os

# Init parser
import configparser
from alodataset import BaseDataset, Split, SequenceMixin, SplitMixin

import aloscene


class Mot17(BaseDataset, SequenceMixin, SplitMixin):

    SPLIT_FOLDERS = {Split.VAL: "train", Split.TRAIN: "train", Split.TEST: "test"}

    SEQUENCES = ["MOT17-05", "MOT17-02", "MOT17-10", "MOT17-13", "MOT17-04", "MOT17-09", "MOT17-11"]

    DPM_SEQUENCES = [
        "MOT17-05-DPM",
        "MOT17-02-DPM",
        "MOT17-10-DPM",
        "MOT17-13-DPM",
        "MOT17-04-DPM",
        "MOT17-09-DPM",
        "MOT17-11-DPM",
    ]

    SDP_SEQUENCES = [
        "MOT17-05-SDP",
        "MOT17-02-SDP",
        "MOT17-10-SDP",
        "MOT17-13-SDP",
        "MOT17-04-SDP",
        "MOT17-09-SDP",
        "MOT17-11-SDP",
    ]

    FRCNN_SEQUENCES = [
        "MOT17-05-FRCNN",
        "MOT17-04-FRCNN",
        "MOT17-09-FRCNN",
        "MOT17-11-FRCNN",
        "MOT17-02-FRCNN",
        "MOT17-10-FRCNN",
        "MOT17-13-FRCNN",
    ]

    def __init__(
        self,
        validation_sequences: list = None,
        training_sequences: list = None,
        detections_set=["DPM", "SDP", "FRCNN"],
        all_gt: bool = False,
        random_step: int = None,
        visibility_threshold=0.0,
        **kwargs
    ):
        """Mot17 Dataset
        https://motchallenge.net/data/MOT17/


        Parameters
        ----------
        mode: str
            "train" or "test"
        validation_sequences: One of MOT17-XX where XX is one of (05, 04, 09, 11, 02, 10, 13)
        all_gt: If True, the confidence score of the boxes will be attached to the boxes.
                Otherwise, only the conf with score==1 will be include,.
        random_step: int
            None by default. If provided, the frame at t+x will be sample with x sampled in between
            (`-random_step`, `random_step`)
        mot_sequences: list
            None by default. If not note, only the provided sequences name will be loaded from the folder
            (folder might change if mode == "test")

        """
        super(Mot17, self).__init__(name="mot17", **kwargs)
        if self.sample:
            return

        if validation_sequences is None:
            raise Exception("validation_sequences must be not None.")
        if all_gt is True:
            raise Exception("TODO: Add the conf score label to the boxes")

        self.random_step = random_step
        self.mot_folder = os.path.join(self.dataset_dir, "train" if self.split is not Split.TEST else "test")
        self.visibility_threshold = visibility_threshold

        self.mot_sequences = {}
        self.items = {}
        listdir = os.listdir(self.mot_folder)

        listdir = []
        if "DPM" in detections_set:
            listdir += self.DPM_SEQUENCES
        if "SDP" in detections_set:
            listdir += self.SDP_SEQUENCES
        if "FRCNN" in detections_set:
            listdir += self.FRCNN_SEQUENCES

        if self.split == Split.TRAIN:
            if training_sequences is not None:
                listdir = [dname for dname in listdir if any([trainseq in dname for trainseq in training_sequences])]
            listdir = [dname for dname in listdir if not any([valseq in dname for valseq in validation_sequences])]
        else:
            assert len([dname for dname in listdir if any([valseq in dname for valseq in validation_sequences])]) > 0
            listdir = [dname for dname in listdir if any([valseq in dname for valseq in validation_sequences])]

        for sequence in listdir:

            config = configparser.ConfigParser()
            config.read(os.path.join(self.mot_folder, sequence, "seqinfo.ini"))

            with open(os.path.join(self.mot_folder, sequence, "gt/gt.txt")) as gt:
                for line in gt.read().split("\n"):
                    if len(line) > 0:
                        self._add_line(sequence, line, config)

            seqs = more_itertools.windowed(
                range(len(self.mot_sequences[sequence])), self.sequence_size, step=1 + self.sequence_skip
            )

            # Update the sequence list
            self.items.update(
                {len(self.items) + idx: {"seq": seq, "mot_sequence": sequence} for idx, seq in enumerate(seqs)}
            )

    def _add_line(self, sequence: str, line: str, config: configparser.ConfigParser):
        """Given a sequence name and the GF `line`
        this method aggrege the sequence data
        """
        if sequence not in self.mot_sequences:
            self.mot_sequences[sequence] = {}

        # print(line)

        frame_id, object_id, box_left, box_top, box_width, box_height, conf, aa, visible = line.split(",")

        frame_id = int(frame_id)
        object_id = int(object_id)

        try:
            self.mot_sequences[sequence][frame_id]
        except:
            self.mot_sequences[sequence][frame_id] = []

        if float(visible) <= self.visibility_threshold:
            return

        conf = float(conf)
        if conf != 1:
            return

        # Boxes coordinates
        box_left = float(box_left)
        box_top = float(box_top)
        box_height = float(box_height)
        box_width = float(box_width)

        self.mot_sequences[sequence][frame_id].append(
            {
                "xc": (box_left + (box_width / 2)) / float(config["Sequence"]["imWidth"]),
                "yc": (box_top + (box_height / 2)) / float(config["Sequence"]["imHeight"]),
                "width": box_width / float(config["Sequence"]["imWidth"]),
                "height": box_height / float(config["Sequence"]["imHeight"]),
                # "frame_size": (int(), int(config["Sequence"]["imWidth"])),
                "object_id": object_id,
            }
        )

    def getitem(self, idx):
        if self.sample:
            return BaseDataset.__getitem__(self, idx)

        seq = list(self.items[idx]["seq"])
        sequence_name = self.items[idx]["mot_sequence"]

        frames = []

        if self.random_step is not None:
            max_seq_len = len(self.mot_sequences[sequence_name]) - 2
            random_step = min((max_seq_len - seq[0]) // self.sequence_size, np.random.randint(1, self.random_step))
            current = seq[0]
            for n in range(1, len(seq)):
                current += random_step
                seq[n] = current

        for s in seq:
            s += 1
            # print("self.split", self.split)
            # print("self.get_split_folder()", self.get_split_folder())
            # print("self.SPLIT_FOLDERS", self.SPLIT_FOLDERS)
            # print("self.SPLIT_FOLDERS[]", self.SPLIT_FOLDERS[self.split])
            image_path = os.path.join(
                self.dataset_dir, self.get_split_folder(), sequence_name, "img1", str(s).zfill(6) + ".jpg"
            )
            n_frame = aloscene.Frame(image_path)

            # Append boxes
            boxes = []
            objects_id = []  # Object Id
            objects_class = []  # Always 1 (Human)

            for data in self.mot_sequences[sequence_name][s]:
                boxes.append([data["xc"], data["yc"], data["width"], data["height"]])
                objects_id.append(data["object_id"])
                objects_class.append(0)  # Human label (Only one label in this dataset)

            # Setup boxes
            if len(boxes) == 0:
                boxes = np.zeros((0, 4))
            boxes = aloscene.BoundingBoxes2D(boxes, boxes_format="xcyc", absolute=False)
            # Setup boxes labels
            objects_class = aloscene.Labels(objects_class, labels_names=["person"], encoding="id", names=("N"))
            objects_id = aloscene.Labels(objects_id, encoding="id", names=("N"))
            # Append labels to the boxes
            boxes.append_labels(objects_class, "objects_class")
            boxes.append_labels(objects_id, "objects_id")
            # Append boxes to the frame
            n_frame.append_boxes2d(boxes)
            # Append the frame to the sequence of frame
            frames.append(n_frame.temporal())

        frames = torch.cat(frames, dim=0)
        return frames


def main():
    """Main"""
    mot_dataset = Mot17(sample=True)
    for frames in mot_dataset.stream_loader():
        frames.names
        frames.get_view(
            [
                frames.boxes2d[0].get_view(frames[0], labels_set="objects_id"),
                frames.boxes2d[1].get_view(frames[1], labels_set="objects_id"),
            ],
            size=(700, 1000),
        ).render()


if __name__ == "__main__":
    main()
