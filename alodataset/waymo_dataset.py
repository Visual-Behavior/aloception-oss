from typing import Dict, List, Tuple
import more_itertools
import pickle as pkl
import numpy as np
import torchvision
import torch
import os


from alodataset import BaseDataset, Split, SequenceMixin, SplitMixin
import aloscene
from aloscene import Labels, Frame, BoundingBoxes2D, BoundingBoxes3D
from aloscene.camera_calib import CameraExtrinsic, CameraIntrinsic

waymo2alo = torch.tensor([[0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])


class WaymoDataset(BaseDataset, SequenceMixin, SplitMixin):

    LABELS = ["gt_boxes_2d", "gt_boxes_3d", "camera_parameters", "traffic_lights"]
    CAMERAS = ["front", "front_left", "front_right", "side_left", "side_right"]
    CLASSES = ["UNKNOWN", "VEHICLE", "PEDESTRIAN", "SIGN", "CYCLIST"]

    SPLIT_FOLDERS = {Split.VAL: "validation", Split.TRAIN: "training", Split.TEST: "testing"}

    def __init__(
        self,
        segments: list = None,
        cameras: list = None,
        random_step: int = None,
        labels: List[str] = [],
        load_rescaled: float = None,
        **kwargs,
    ):
        """WaymoDataset

        Parameters
        ----------
        segments: list or None
            List waymo segments to load. If None, all segments will be loaded.
        cameras: list or None
            List of camera to use. If none, all cameras data will be loaded
            List could be some of ["front", "front_left", "front_right", "side_left", "side_right", "all"].
            If "all" is selected, data from cameras will be merge independently of the source.
        random_step: int
            None by default. Otherwise, sample t+1 randomly on each sequence.
        labels: list of strings
            List could be some of ["gt_boxes_2d", "gt_boxes_3d", "camera_parameters", "traffic_lights"]
        load_rescaled: float
            Downscale factor of images to load (>1). If None, standard resoltion will be loaded (1280,1920).
            There should a camera dictory with the rescaled images.

        """
        super(WaymoDataset, self).__init__(name="waymo", **kwargs)
        if "traffic_lights" in labels:
            self.CLASSES.append("TRAFFIC_LIGHTS")
        if self.sample:
            self.cameras = self.CAMERAS
            return

        self.random_step = random_step
        self.load_rescaled = load_rescaled

        # Will be filled into the load() method
        self.preloaded_labels_2d = {}
        self.preloaded_labels_3d = {}
        self.preloaded_calib = {}
        self.preloaded_traffic_lights = {}

        self.labels = labels
        self.cameras = cameras if cameras is not None else self.CAMERAS
        # Idx to sequences
        self.items = {}
        self.segment_size = {}

        self.dataset_dir = os.path.normpath(self.dataset_dir)

        self.segments = segments
        if "_prepared" in self.dataset_dir:
            self.read_sequences()

    def get_segments(self) -> List[str]:
        """Read dataset directory based on `self.dataset_dir` and `self.split`.
        Return a list of segment names found in that directory.

        Returns
        -------
        List[str]
            List of segment names
        """
        seg_dir = os.path.join(self.dataset_dir, self.get_split_folder())
        segments = [
            d for d in os.listdir(seg_dir) if os.path.isdir(os.path.join(seg_dir, d)) and d.startswith("segment")
        ]
        return segments

    def read_sequences(self):
        """Populate all dictionaries used internally by this class to manage sequences, segments, etc.

        Dictionaries:
            - `self.items`:
                - key: int, item id
                - value: dict of
                    - "sequence" : tuple of sequence id
                    - "segment" : str, segment name

            - `self.preloaded_labels_2d`:
                - key: str, segment name
                - value: dict
                    - key: sequence id
                    - value: dict of
                        - key: str, camera id
                        - value: list of labels as dict of keys "bbox", "track_id", "class", "camera_id"

            - `self.preloaded_labels_3d`:
                - key: str, segment name
                - value: dict
                    - key: sequence id
                    - value: dict of
                        - key: str, camera id
                        - value: list of labels as dict of keys "bbox_proj", "bbox_3d", "track_id", "class",
                                "camera_id", "speed", "accel"

            - `self.preloaded_calib`:
                - key: str, segment name
                - value: dict
                    - key: sequence id
                    - value: dict of
                        - key: "cam_intrinsic", "cam_extrinsic"
                        - value: dict of:
                            - key: int, camera id + 1
                            - value: np.ndarray of shape (3, 4) for cam_intrisic or shape (4, 4) for cam_extrinsic
        """
        self.segments = self.segments if self.segments is not None else self.get_segments()
        for segment in self.segments:
            segment_folder = os.path.join(self.dataset_dir, self.get_split_folder(), segment)
            # Load Boxes 2D labels
            if "gt_boxes_2d" in self.labels:
                with open(os.path.join(segment_folder, "camera_label.pkl"), "rb") as f:
                    self.preloaded_labels_2d[segment] = pkl.load(f)
            if "traffic_lights" in self.labels:
                with open(os.path.join(segment_folder, "traffic_lights_label.pkl"), "rb") as f:
                    self.preloaded_traffic_lights[segment] = pkl.load(f)
            # Load boxes 3D labels
            if "gt_boxes_3d" in self.labels:
                with open(os.path.join(segment_folder, "lidar_label.pkl"), "rb") as f:
                    self.preloaded_labels_3d[segment] = pkl.load(f)
            # Load cameras parameters
            if "camera_parameters" in self.labels or "gt_boxes_3d" in self.labels:
                with open(os.path.join(segment_folder, "calib.pkl"), "rb") as f:
                    self.preloaded_calib[segment] = pkl.load(f)

            # Create the sequence ids
            if self.load_rescaled:
                ids = os.listdir(os.path.join(segment_folder, f"image0:{self.load_rescaled:.1f}"))
            else:
                ids = os.listdir(os.path.join(segment_folder, "image0"))
            num_step = max(int(el_id[0:3]) for el_id in ids)
            self.segment_size[segment] = num_step

            if "all" in self.cameras:
                for _ in range(len(self.CAMERAS)):
                    sequences = more_itertools.windowed(
                        range(num_step), self.sequence_size, step=self.sequence_skip + 1
                    )
                    # Update the sequence list
                    self.items.update(
                        {
                            len(self.items) + idx: {"sequence": sequence, "segment": segment}
                            for idx, sequence in enumerate(sequences)
                        }
                    )
            else:
                sequences = more_itertools.windowed(range(num_step), self.sequence_size, step=self.sequence_skip + 1)
                # Update the sequence list
                self.items.update(
                    {
                        len(self.items) + idx: {"sequence": sequence, "segment": segment}
                        for idx, sequence in enumerate(sequences)
                    }
                )

    def get_frame_boxes2d(
        self, frame: Frame, camera: str, segment: str, sequence_id: int, idstring2int
    ) -> BoundingBoxes2D:
        """Parse preloaded dict and return BoundingBoxes2D

        Parameters
        ----------
        frame : Frame
        camera : str
            Camera name. One of ["front", "front_left", "front_right", "side_left", "side_right"]
        segment : str
            Segment name
        sequence_id : int
            Sequence id in segment
        idstring2int : dict
            - key: str, track id string
            - value: int, unique value represents a track id string

            `idstring2int` can be empty. A mapping of id string and int will be created on the fly.

        Returns
        -------
        BoundingBoxes2D
            Shape (n, 4). Each box is associtated with label and track id
        """
        camera_id = str(self.CAMERAS.index(camera))

        anns = self.preloaded_labels_2d[segment][int(sequence_id)][camera_id]

        if "traffic_lights" in self.labels and self.preloaded_traffic_lights[segment][int(sequence_id)] is not None:
            # add trafic lights annotations
            anns = anns + self.preloaded_traffic_lights[segment][int(sequence_id)][int(camera_id)]

        boxes = np.zeros((len(anns), 4))
        labels = np.zeros((len(anns)))
        track_id = np.zeros((len(anns)))

        for a, ann in enumerate(anns):

            boxes[a] = np.array(ann["bbox"])
            labels[a] = ann["class"]
            # track_id.append(ann["track_id"])
            if ann["track_id"] is not None:
                try:
                    tid = idstring2int[ann["track_id"]]
                except:
                    idstring2int[ann["track_id"]] = len(idstring2int) + 1
                    tid = idstring2int[ann["track_id"]]
            else:
                tid = -1
            track_id[a] = tid

        # boxes = np.expand_dims(boxes, axis=0)
        labels_2d = Labels(labels, encoding="id", labels_names=self.CLASSES, names=("N"))
        track_id = Labels(track_id, encoding="id", names=("N"))
        if self.load_rescaled:
            frame_size = (frame.H * self.load_rescaled, frame.W * self.load_rescaled)
        else:
            frame_size = frame.HW
        boxes_2d = BoundingBoxes2D(
            boxes,
            boxes_format="xcyc",
            absolute=True,
            names=("N", None),
            frame_size=frame_size,
            labels={"class": labels_2d, "track_id": track_id},
        )
        if "traffic_lights" in self.labels:
            frame.append_labels(
                Labels(
                    torch.tensor([self.preloaded_traffic_lights[segment][int(sequence_id)] is not None]), names=(None,)
                ),
                "traffic_lights_annotated",
            )
        return boxes_2d

    def get_frame_camera_parameters(
        self, frame: Frame, camera: str, segment: str, sequence_id: int
    ) -> Tuple[CameraIntrinsic, CameraExtrinsic]:
        """Parse preloaded dict and return a tuple of CamIntrinsic and CamExtrinsic

        Parameters
        ----------
        frame : Frame
        camera : str
            Camera name. One of ["front", "front_left", "front_right", "side_left", "side_right"]
        segment : str
            Segment name
        sequence_id : int
            Sequence id in segment

        Returns
        -------
        Tuple[CameraIntrinsic, CameraExtrinsic]
            - CameraIntrinsic: shape (3, 4)
            - CameraExtrinsic: shape (4, 4)
        """
        np_cam_intrinsic = self.preloaded_calib[segment][int(sequence_id)]["cam_intrinsic"][
            self.CAMERAS.index(camera) + 1
        ]
        cam_intrinsic = torch.Tensor(np_cam_intrinsic)
        cam_intrinsic = CameraIntrinsic(cam_intrinsic[None], names=("T", None, None))

        np_cam_extrinsic = self.preloaded_calib[segment][int(sequence_id)]["cam_extrinsic"][
            self.CAMERAS.index(camera) + 1
        ]
        np_cam_extrinsic = np.linalg.inv(np_cam_extrinsic)
        cam_extrinsic = torch.Tensor(np_cam_extrinsic)
        # transform cam_extrinsic in aloception coordinate system
        cam_extrinsic = waymo2alo @ cam_extrinsic @ torch.linalg.inv(waymo2alo)
        cam_extrinsic = CameraExtrinsic(cam_extrinsic[None], names=("T", None, None))

        return cam_intrinsic, cam_extrinsic

    @staticmethod
    def np_convert_waymo_to_aloception_coordinate_system(np_boxes3d: np.ndarray) -> np.ndarray:
        """Transform boxes3d from waymo coordinates to aloception coordinates

        Waymo coordinates:
            - X forward
            - Y left
            - Z upward

        Aloception coordinates:
            - X right
            - Y downward
            - Z forward

        Parameters
        ----------
        np_boxes3d : np.ndarray
            boxes 3d in waymo coordinates, shape (n, 7)

        Returns
        -------
        np.ndarray
            boxes 3d in aloception coordinates, shape (n, 7)
        """
        # swap axis
        center = waymo2alo[:3, :3] @ np_boxes3d[:, :3].T  # (3, n)
        center = center.T  # (n, 3)
        # swap dimension
        dimension = np_boxes3d[:, [4, 5, 3]]
        # inverse heading
        heading = -np_boxes3d[:, 6:7]
        np_boxes3d = np.concatenate([center, dimension, heading], axis=-1)
        return np_boxes3d

    def get_frame_boxes3d(self, frame, camera, segment, sequence_id) -> BoundingBoxes3D:
        """Parse preloaded dict and return BoundingBoxes3D

        Parameters
        ----------
        frame : Frame
        camera : str
            Camera name. One of ["front", "front_left", "front_right", "side_left", "side_right"]
        segment : str
            Segment name
        sequence_id : int
            Sequence id in segment

        Returns
        -------
        BoundingBoxes3D
            boxes 3d, shape (n, 7)
        BoundingBoxes2D
            boxes 3d projected on image, shape (n, 4)
        """
        camera_id = str(self.CAMERAS.index(camera))
        anns = self.preloaded_labels_3d[segment][int(sequence_id)][camera_id]

        np_boxes3d = np.zeros((len(anns), 7))
        np_boxes3d_proj = np.zeros((len(anns), 4))
        np_labels = np.zeros((len(anns),))
        track_id = []
        for a, ann in enumerate(anns):
            np_boxes3d[a] = np.array(ann["bbox_3d"])
            np_boxes3d_proj[a] = np.array(ann["bbox_proj"])
            np_labels[a] = ann["class"]
            track_id.append(ann["track_id"])
        np_boxes3d = self.np_convert_waymo_to_aloception_coordinate_system(np_boxes3d)
        labels = Labels(np_labels, names=("N"), encoding="id", labels_names=self.CLASSES)
        boxes3d = BoundingBoxes3D(np_boxes3d, labels=labels, names=("N", None))
        if self.load_rescaled:
            frame_size = (frame.H * self.load_rescaled, frame.W * self.load_rescaled)
        else:
            frame_size = frame.HW

        boxes3d_proj = BoundingBoxes2D(
            np_boxes3d_proj,
            boxes_format="xcyc",
            absolute=True,
            names=("N", None),
            frame_size=frame_size,
            labels=labels,
        )
        return boxes3d, boxes3d_proj

    def get_frames(self, camera: str, segment: str, sequence: list) -> aloscene.Frame:
        """Get a tensor Frame given a camera, a segment name and the list of sequence ids

        Parameters
        ----------
        camera : str
            Camera name. One of ["front", "front_left", "front_right", "side_left", "side_right"]
        segment : str
            Segment name
        sequence : list of int
            List of sequence id

        Returns
        -------
        aloscene.Frame
            Frame contain ground truth boxes 2d, 3d with labels and camera intrinsic/extrinsic matrix
            based on `self.labels` set at `self.__init__`
        """
        camera_id = str(self.CAMERAS.index(camera))
        idstring2int = {}
        frames = []
        t = 0

        if self.random_step is not None and len(sequence) > 1:
            sequence = list(sequence)
            current = sequence[0]
            random_step = min(
                (self.segment_size[segment] - sequence[0]) // self.sequence_size,
                np.random.randint(1, self.random_step),
            )
            for t in range(1, len(sequence)):
                current += random_step
                sequence[t] = current
            sequence = tuple(sequence)

        for el in sequence:
            # Open image
            if self.load_rescaled:
                img_camera_id = f"{camera_id}:{self.load_rescaled:.1f}"
            else:
                img_camera_id = camera_id
            image_path = os.path.join(
                self.dataset_dir, self.get_split_folder(), segment, "image" + img_camera_id, str(el).zfill(3) + ".jpg"
            )
            image = torchvision.io.read_image(image_path)
            # Add the sequence dimension
            image = torch.unsqueeze(image, dim=0)
            frame = Frame(image.type(torch.float32), normalization="255", names=("T", "C", "H", "W"))
            if "gt_boxes_2d" in self.labels:
                boxes_2d = self.get_frame_boxes2d(frame, camera, segment, el, idstring2int)
                # if t == 0:
                frame.append_boxes2d(boxes_2d, "gt_boxes_2d")
            if "gt_boxes_3d" in self.labels:
                boxes_3d, boxes_3d_proj = self.get_frame_boxes3d(frame, camera, segment, el)
                frame.append_boxes3d(boxes_3d, "gt_boxes_3d")
                frame.append_boxes2d(boxes_3d_proj, "gt_boxes_3d_proj")
            if "camera_parameters" in self.labels or "gt_boxes_3d" in self.labels:
                cam_intrinsic, cam_extrinsic = self.get_frame_camera_parameters(frame, camera, segment, el)
                frame.append_cam_intrinsic(cam_intrinsic)
                frame.append_cam_extrinsic(cam_extrinsic)

            if self.load_rescaled:

                def resize_func(label):
                    # resize with relative coordinates if possible, else return unmodified label
                    try:
                        label_resized = label._resize((1.0 / self.load_rescaled, 1.0 / self.load_rescaled))
                        return label_resized
                    except AttributeError:
                        return label

                frame = frame.recursive_apply_on_children_(resize_func)

            if "depth" in self.labels:
                depth_path = os.path.join(
                    self.dataset_dir, self.get_split_folder(), segment, "depth" + camera_id, str(el).zfill(3) + ".npz"
                )
                depth = aloscene.Depth(depth_path).temporal()

                # depth is at 1/4th the resolution so adapt image
                if frame.HW != depth.HW:
                    frame = frame.resize(depth.HW)
                frame.append_depth(depth)

            frames.append(frame)
            t += 1

        # Stack along the first dimension
        frames = torch.cat(frames, dim=0).type(torch.float32)

        return frames

    def getitem(self, idx) -> Dict[str, aloscene.Frame]:
        """Given item id, get a dict of which keys are camera name and value are a Frame with ground truth

        Parameters
        ----------
        idx : int
            Item id, see `self.read_sequences()` for more details.

        Returns
        -------
        Dict[str, aloscene.Frame]
            - key: str, camera name
            - value: aloscene.Frame containing labels 2d, 3d and camera intrinsic/extrinsic
                    based on `self.labels` set by `self.__init__`
        """
        if self.sample:
            return BaseDataset.__getitem__(self, idx)

        sequence_data = self.items[idx]
        data = {}
        for o_id, camera in enumerate(self.cameras):
            if camera == "all":
                # Choose a camera at random among the possible cameraw
                sampled_camera = self.CAMERAS[idx % len(self.CAMERAS)]
                frames = self.get_frames(sampled_camera, sequence_data["segment"], sequence_data["sequence"])
            else:
                frames = self.get_frames(camera, sequence_data["segment"], sequence_data["sequence"])
            data[camera] = frames
        return data

    def prepare(self, num_processes=2, depth=False):
        """
        Prepre Waymo Open Dataset from tfrecord files.
        The preparation can be resumed if it was stopped suddenly.

        The expected tree:\n
            waymo_data_dir\n
            \\|__validation\n
            \\|__training\n
            \\|__test

        Each subdirectory must contains tfrecord files extracted from Waymo Open Dataset tar files.

        Read TFRecord files recursively in `self.dataset_dir` and prepare pickle files and images
        in `self.dataset_dir + "_prepare"` directory. Once the dataset is all prepared, the path
        to the dir in /.aloception/alodataset_config.json will be replace by the new prepared one.
        If `self.dataset_dir` aldready ends with "_prepare", this function does nothing.


        Please install alodataset/prepare/waymo-requirements.txt

        Notes
        -----
        If the dataset is already prepared, this method will simply check that all file
        are prepared and stored into the prepared folder. If the original directory is no longer
        on the disk, the method will simply use the prepared dir as it is and the prepare step will be skiped.
        """

        if self.sample:
            return

        if self.dataset_dir.endswith("_prepared") and not os.path.exists(self.dataset_dir.replace("_prepared", "")):
            return

        from alodataset.prepare.waymo_converter import Waymo2KITTIConverter

        # If dataset_dir is not prepared, prompt user permission
        if not self.dataset_dir.endswith("_prepared"):
            ans = input(
                "The prepared Waymo dataset will take up 400GB of disk space. Do you want to continue? (Y/N): "
            )
            if ans.lower() != "y":
                raise Exception("Waymo dataset preparation is stopped by user")

        # Create wip dir and prepared dir
        dataset_dir = (
            self.dataset_dir
            if not self.dataset_dir.endswith("_prepared")
            else self.dataset_dir.replace("_prepared", "")
        )
        dataset_root, dataset_dir_name = os.path.split(dataset_dir)
        subdir_names = [name for name in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, name))]
        prepared_dir_name = f"{dataset_dir_name}_prepared"
        prepared_dir = os.path.join(dataset_root, prepared_dir_name)
        if not os.path.exists(prepared_dir):
            os.makedirs(prepared_dir)

        # For each subdir (training, validation, testing) in self.dataset_dir
        # the conversion can be resumed if it was stopped suddenly
        for subdir_name in subdir_names:
            print(f"\nPreparing {subdir_name} ...")
            prepared_subdir = os.path.join(prepared_dir, subdir_name)
            subdir = os.path.join(dataset_dir, subdir_name)
            if not os.path.exists(prepared_subdir):
                os.makedirs(prepared_subdir)
            Waymo2KITTIConverter(subdir, prepared_subdir, num_proc=num_processes, depth=depth).convert()
            print(f"{subdir_name} prepared.")

        # Set new dataset_dir as prepared_dir in aloception.config
        self.set_dataset_dir(prepared_dir)
        self.read_sequences()


def main():
    """Main"""
    waymo_dataset = WaymoDataset(labels=["gt_boxes_2d", "gt_boxes_3d", "depth", "traffic_lights"], load_rescaled=4.0)
    # waymo_dataset.prepare()

    for frames in waymo_dataset.train_loader(batch_size=2):
        frames = Frame.batch_list([frame["front"] for frame in frames])
        frames.get_view([frames.boxes3d, frames.boxes2d]).render()


if __name__ == "__main__":
    main()
