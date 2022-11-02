from aloscene import Frame, Mask, BoundingBoxes2D, Labels
from alodataset import BaseDataset

from PIL import Image
import numpy as np
import torch
import glob
import os


class WooodScapeDataset(BaseDataset):
    """WoodScape dataset iterator

    Paramneters
    -----------
        labels : List[str]
            List of labels to stick to the frame. If the list is empty all labels are loaded. By default all labels are attached.
        cameras : List[str]
            List of cameras to consider. If the list empty all cameras are loaded. By default all camera views are considered.
        fragment : Union[int, float]
            Either the portion of dataset to to consider if the arg is float or the number of samples if int.
            Passing a negative value will start the count from the end. By default 0.9
        seg_classes : List[sstr]
            Classes to consider for segmentation. By default all classes are considered.
        merge_classees : bool
            Assign the same classe index for all segementation classes, Default if False.
        rename_merged : str
            Name to give to merged instancee. Only if merge_classes is True. Default is "mix".

    Raises
    ------
        AssertionError
            One of the labels is not in ["Seg", "bxox_2d"].
        AssertionError
            One of the cameras is not in ["LV", "FV", "MVL", "MVR"].
        AssertionError
            One of the passed clases is not available/supported.

    """

    CAMERAS = [
        "RV",  # Right View
        "FV",  # Front View
        "MVL",  # Mirror Left View
        "MVR",  # Mirror Right View
    ]
    LABELS = ["seg", "box_2d"]
    SEG_CLASSES = [
        "void",
        "road",
        "lanemarks",
        "curb",
        "person",
        "rider",
        "vehicles",
        "bicycle",
        "motorcycle",
        "traffic_sign",
    ]

    def __init__(
        self,
        labels=[],
        cameras=[],
        fragment=0.9,
        name="WoodScape",
        seg_classes=[],
        merge_classes=False,
        rename_merged="mix",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        cameras = self.CAMERAS if cameras == list() else cameras
        labels = self.LABELS if labels == list() else labels

        if isinstance(fragment, int):
            pass
        elif isinstance(fragment, float):
            assert fragment <= 1 and fragment >= -1, "fragment of type float can not be higher than 1 of less than -1."
        else:
            raise AttributeError("Invalid type of fragment type.")

        if seg_classes == list():
            seg_classes = self.SEG_CLASSES

        assert all(
            [c in self.SEG_CLASSES for c in seg_classes]
        ), f"some segmentation classes are invalid, supported classes are :\n {self.SEG_CLASSES}"
        assert all([v in self.CAMERAS for v in cameras]), f"Some cameras are invalid, should be in {self.CAMERAS}"
        assert all([l in self.LABELS for l in labels]), f"Some labels are invalid, should be ib {self.LABELS}"
        assert isinstance(merge_classes, (int, bool)), "Invalid merge_classes argument"
        assert isinstance(rename_merged, str), "Invalid rename_merged argument"

        self.items = sorted(glob.glob(os.path.join(self.dataset_dir, "rgb_images", "*")))
        self.items = self._filter_cameras(self.items, cameras)
        self.items = self._filter_non_png(self.items)

        self.labels = labels
        self.cameras = cameras
        self.seg_classes = seg_classes
        self.merge_classes = merge_classes
        self.num_seg_classes = len(seg_classes)
        self.seg_classes_renamed = seg_classes if not merge_classes else [rename_merged]

        # Encode fraction
        self.fragment = min(abs(fragment), len(self)) if isinstance(fragment, int) else int(abs(fragment) * len(self))

        # Restricting the number of samples
        if fragment > 0:
            self.items = self.items[: self.fragment]
        else:
            self.items = self.items[len(self) - self.fragment :]

    def getitem(self, idx):
        ipath = self.items[idx]
        frame = Frame(ipath, names=tuple("CHW"))

        if "seg" in self.labels:
            segmentation = self._path2segLabel(ipath)
            frame.append_segmentation(segmentation)

        if "box_2d" in self.labels:
            _, H, W = frame.shape
            bbox2d_path = self._path2boxLabel(ipath, frame_size=(H, W))
            frame.append_boxes2d(bbox2d_path)
        return frame

    @staticmethod
    def _path2segPath(path):
        """Maps rgb image path to corresponding segmentation path

        Parameters
        ----------
            path: str
                Path to rgb image.

        """
        path, file = os.path.split(path)
        path, _ = os.path.split(path)
        path = os.path.join(path, "semantic_annotations", "gtLabels", file)
        return path

    @staticmethod
    def _path2boxPath(path):
        """Maps rgb image path to corresponding json 2dbbox file

        Parameters
        ----------
            path: str
                path to rgb image

        """
        path, file = os.path.split(path)
        path, _ = os.path.split(path)
        path = os.path.join(path, "box_2d_annotations", file.replace(".png", ".txt"))
        return path

    def _path2segLabel(self, path):
        """Maps image path to segmentation mask

        Parametrs
        ---------
            path: str
                path to rgb image

        """
        path = self._path2segPath(path)
        mask = np.asarray(Image.open(path))
        mask = self.mask_2d_idx_to_3d_onehot_mask(mask)
        return mask

    def mask_2d_idx_to_3d_onehot_mask(self, mask_2d):
        """Converts 2d index encoding mask to 3d one hot encoding one

        Parameters
        ----------
            mask : np.ndarray
                Mask of size (H, W) with int values

        """
        sample_seg_classes = torch.unique(torch.Tensor(mask_2d.reshape(-1)))

        num_sample_seg_classes = len(self.seg_classes_renamed)
        mask_3d = np.zeros((num_sample_seg_classes,) + mask_2d.shape)

        dec = 0
        for i, name in enumerate(self.seg_classes):
            if i in sample_seg_classes:
                mask_3d[i - dec] += (mask_2d == self.SEG_CLASSES.index(name)).astype(int)
                if self.merge_classes:
                    dec += 1
            else:
                dec += 1

        mask_3d = Mask(mask_3d, names=tuple("CHW"))
        mlabels = Labels(
            torch.arange(num_sample_seg_classes).to(torch.float32),
            labels_names=self.seg_classes_renamed,
            names=("N"),
            encoding="id",
        )
        mask_3d.append_labels(mlabels)
        return mask_3d

    def _path2boxLabel(self, path, frame_size):
        """Maps image patgh to bbox2d label

        Parameters
        ----------
            path: str
                rgb image path

        """
        path = self._path2boxPath(path)

        with open(path, "r") as f:
            content = f.readlines()
        content = [x.replace("\n", "") for x in content]
        content = [x.split(",") for x in content]
        bboxs2d = [[int(x) for x in c[2:]] for c in content]
        return BoundingBoxes2D(bboxs2d, boxes_format="xyxy", absolute=True, frame_size=frame_size)

    @staticmethod
    def _filter_non_png(items):
        """Filters non png files from a list of paths to files

        Parameters
        ----------
            items : List[str]
                list of paths to filter

        """
        return [p for p in items if p.endswith(".png")]

    @staticmethod
    def _filter_cameras(items, cameras):
        """Filters paths by given cameras list

        Parameters
        ----------

            cameras : List[str]
                List of cameras

        """
        return list(filter(lambda x: any([v in x for v in cameras]), items))


if __name__ == "__main__":
    ds = WooodScapeDataset(
        labels=[],
        cameras=[],
        fragment=1.0,
    )
    idx = 222
    frame = ds[idx]
    frame.get_view().render()
