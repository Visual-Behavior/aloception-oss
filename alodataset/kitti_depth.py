import os
import glob
import warnings
import numpy as np
from PIL import Image
from typing import List, Dict, Union
from aloscene import Frame, Depth, CameraIntrinsic, AugmentedTensor
from alodataset import BaseDataset
from alodataset.utils.kitti import load_calib_cam_to_cam


class KittiDepth(BaseDataset):
    """
    Depth task from KITTI dataset.
    Parameters
    ----------
        subset: either 'train', 'val' or 'all'. If all, both val and train subset are considered.

        return_depth: Either to include depth in the output or not.

        custom_drives: Dictionnary (Keys (:str) registration date of the drive ('YYYY_MM_DD'). Values: (:List[Union[str, Dict]])
                       either ids of drives or dictionnaries of drives of ids and specific files.
                       Exemples: custom_drives = {'2011_09_28': ['0222', '0186']} will load all images from
                                 2011_09_28_drive_0222_sync and 2011_09_28_drive_0186_sync.
                                 custom_drives = {'2011_09_28': {'0222': ['0000000015', '0000000010']}} will load the two images.
                       Ids should match the drives dates, otherwise the drive folder will be ignored.
                       If None, all images from specified subset are loaded.

        name: Key of database name in :attr:`alodataset_config.json` file, by default *kitti*.

        main_folder: The main folder of input images, could be image_00, image_01, image_02 or image_03.

        sample: Download (or not) a dataset sample from internet and replace the default dataset_dir.


    Attributes
    ----------
        drives: List of paths to kitti drive folders.

        drives_folders: List of paths to drives' parent forlders.


    Exemples
    --------
        >>> # get all the dataset
        >>> kitti_ds = KittiDepth(
                                subset='all',
                                return_depth=True,
                                )
        >>> # get specific files from both val and train
        >>> idsOfDrives = ['0001',  # drive from the training subset
                           '0002',  # drive from the validation subset
                          ]
        >>> date = '2011_09_26'
        >>> custom_drives = {date: idsOfDrives}
        >>> kitti_ds = KittiDepth(
                                return_depth=True,
                                custom_drives=custom_drives,
                                )
    """

    VALID_SUBSETS = ["train", "val", "all"]
    VALID_MAIN_FOLDERS = ["image_00", "image_01", "image_02", "image_03"]

    def __init__(
        self,
        subset: str = "all",
        return_depth: bool = True,
        return_ground: bool = False,
        custom_drives: Union[Dict[str, List[str]], None] = None,
        name: str = "kitti",
        main_folder: str = "image_02",
        sample: bool = False,
        **kwargs,
    ):
        if sample:
            raise NotImplementedError("Download option is not yet available for kitti.")
        super(KittiDepth, self).__init__(name=name, **kwargs)

        assert main_folder in self.VALID_MAIN_FOLDERS, f"main_folder should be in {self.VALID_MAIN_FOLDERS}"
        assert subset in self.VALID_SUBSETS, f"subset should be one of {self.VALID_SUBSETS}"

        self.main_folder = main_folder
        self.return_depth = return_depth
        self.return_ground = return_ground
        self.drives = list()
        self.drives_folders = list()
        self.subset = subset
        self.custom_drives = custom_drives
        if self.subset == "all":
            self.subset = self.VALID_SUBSETS[:2]
        else:
            self.subset = [self.subset]

        for ss in self.subset:
            self.drives_folders.append(os.path.join(self.dataset_dir, "images", ss))

        if custom_drives is None:
            for dfolder in self.drives_folders:
                self.drives += glob.glob(os.path.join(dfolder, "*"))

        else:
            for date in custom_drives.keys():
                for drive_id in custom_drives[date]:
                    drive_exits = False
                    if isinstance(drive_id, str):
                        drive_name = self.get_drive_name(date, drive_id)
                    elif isinstance(drive_id, dict):
                        drive_name = self.get_drive_name(date, list(drive_id.keys())[0])
                    else:
                        raise AttributeError("invalid custom drive format")

                    for dfolder in self.drives_folders:
                        path_to_drive = os.path.join(dfolder, drive_name)
                        if os.path.isdir(path_to_drive):
                            drive_exits = True
                            self.drives.append(path_to_drive)
                            # break if path exists in the validation subset
                            break
                    if not drive_exits:
                        print(
                            f"Warning: no drive named {drive_name}. Drive id and date do not match or drive is not in the subset"
                        )

        for drive in self.drives:
            if self.custom_drives:
                basename = os.path.basename(drive)
                date, drive_id = self.date_id_from_drive(basename)
                if isinstance(self.custom_drives[date], list):
                    self.items += self._get_paths_from_subdirs(drive)
                if isinstance(self.custom_drives[date], dict):
                    self.items += self._get_paths_from_subdirs(drive, self.custom_drives[date][drive_id])

            else:
                self.items += self._get_paths_from_subdirs(drive)

        # making sure corresponding depth files exist
        if self.return_depth:
            replace_items = list()
            for item in self.items:
                depth_path = self.get_corresponding_depth(item)
                if os.path.exists(depth_path):
                    replace_items.append(item)
            lost_items = len(self.items) - len(replace_items)
            if lost_items > 0:
                print(f"ground truth images not found for {lost_items} images")
                self.items = replace_items

        self.calibs = self._get_calibrations()

        if self.custom_drives and not self.items:
            raise AttributeError("Could not read any images from the given custom drives")

    def _get_calibrations(self):
        calibs = {}
        calibs_dir = os.path.join(self.dataset_dir, "calibs")
        for folder in os.listdir(calibs_dir):
            path = os.path.join(calibs_dir, folder, folder[:10])
            data = load_calib_cam_to_cam(
                os.path.join(path, "calib_cam_to_cam.txt"),
                os.path.join(path, "calib_velo_to_cam.txt"),
            )
            Tr_velo_to_cam = self._get_velo_to_cam(
                os.path.join(path, "calib_velo_to_cam.txt"),
            )
            calibs[folder[:10]] = {
                "intrinsic": np.c_[data["K_cam2"], [0, 0, 0]],
                "extrinsic": data["T_cam2_rect"],
                "ground_matrix": data["P_rect_20"] @ data["R_rect_00"] @ Tr_velo_to_cam,
            }
        return calibs

    def _get_velo_to_cam(self, path):
        with open(path, "r") as f:
            calib_velo_to_cam = f.readlines()
        velo_to_cam_R = np.matrix([float(x) for x in calib_velo_to_cam[1].strip("\n").split(" ")[1:]]).reshape(3, 3)
        Tr_velo_to_cam = np.insert(
            velo_to_cam_R, 3, values=np.array(calib_velo_to_cam[2].strip("\n").split(" ")[1:]), axis=1
        )
        Tr_velo_to_cam = np.insert(Tr_velo_to_cam, 3, values=[0, 0, 0, 1], axis=0)
        return Tr_velo_to_cam

    @classmethod
    def from_yaml(cls, path: str, **kwargs):
        """Pass custom drives from a .yaml file.

        Parameters
        ----------

        path: str
            Path to the .yaml file.

        Returns
        -------
        KittiDepth
            KittiDepth dataset with custom drives.
        """
        import yaml

        with open(path, "r") as f:
            custom_drives = yaml.load(f, Loader=yaml.FullLoader)
        return cls(custom_drives=custom_drives, **kwargs)

    @staticmethod
    def depth_from_path(path: str) -> Depth:
        """Reads depth from the given path.

        Parameters
        ----------
        path: str
            Path to the depth file.

        Returns
        -------
        Depth
            Depth object.
        """
        depth = np.asarray(Image.open(path), dtype=np.int16)
        return Depth(np.expand_dims(depth, 0))

    def getitem(self, idx):
        """Get the :mod:`Frame <aloscene.frame>` corresponds to *idx* index.

        Parameters
        ----------
        idx : int
            Index of the frame to be returned.

        Returns
        -------
        :mod:`Frame <aloscene.frame>`
            Frame with its corresponding depth.
        """
        if self.sample:
            return BaseDataset.__getitem__(self, idx)

        image_path = self.items[idx]

        # add calibrations
        sequence = image_path.split(os.sep)[-4]
        intrinsic = CameraIntrinsic(self.calibs[sequence[:10]]["intrinsic"])

        frame = Frame(image_path, cam_intrinsic=intrinsic)

        if self.return_depth:
            depth_path = self.get_corresponding_depth(image_path)
            depth = self.depth_from_path(depth_path)
            frame.append_depth(depth)

        if self.return_ground:
            frame.add_child(
                "ground_matrix", AugmentedTensor(self.calibs[sequence[:10]]["ground_matrix"], names=(None, None))
            )

        return frame

    def get_corresponding_depth(self, path: str):
        """
        Get the corresponding depth path from the given image path.

        Parameters
        ----------
        path: str
            Path to drive.

        Returns:
        --------
        path: str
            Corresponding path to depth groud truth image.
        """
        fixed_dir = os.path.join("proj_depth", "groundtruth")
        filename = os.path.basename(path)
        driveAndSubset = os.path.join(*path.split(os.sep)[-5:-3])
        return os.path.join(self.dataset_dir, "depth", driveAndSubset, fixed_dir, self.main_folder, filename)

    def _get_paths_from_subdirs(
        self, drive_path: str, filenames: Union[List[str], None] = None, extension: str = "png"
    ) -> List[str]:
        """Return paths to images in drive_path/self.main_folder/data with the specified extension.

        Parameters
        ----------
        drive_path: str
            Path to the drive.
        filenames: list, optional
            List of filenames to be returned. If None, all files are returned.
        extension: str, optional
            Extension of the files to be returned. Default is "png".

        Returns
        -------
        list[str]
            List of paths to images.
        """
        assert extension in ["png", "jpg"], "unvalid extension: should be 'png' | 'jpg'"
        main_path = os.path.join(drive_path, self.main_folder, "data")

        if not os.path.exists(main_path):
            drive = os.path.basename(drive_path)
            print(f"main images folder {self.main_folder} not found for drive {drive}")
        if filenames:
            return [os.path.join(main_path, f"{filename}.{extension}") for filename in filenames]
        else:
            return sorted(glob.glob(os.path.join(main_path, f"*.{extension}")))

    @staticmethod
    def get_drive_name(date: str, drive_id: str) -> str:
        """Return the drive file name with the corresponding date and drive id.

        Parameters
        ----------
        date: str
            Date of the drive.
        drive_id: str
            Id of the drive.

        Returns
        -------
        str
            Drive name."""
        return date + "_drive_" + drive_id + "_sync"

    @staticmethod
    def date_id_from_drive(basename):
        """Return the date and drive id from the drive name.

        Parameters
        ----------
        basename: str
            Drive name.

        Returns
        -------
        str
            Date and id of the drive.
        """
        return basename[:10], basename[17:21]


if __name__ == "__main__":
    # date = "2011_09_26"
    # idsOfDrives = [
    #     "0001",  # sample from training subset
    #     "0002",  # sample from validation subset
    # ]
    # custom_drives = {date: idsOfDrives}
    kitti_ds = KittiDepth(
        # subset="all",
        # return_depth=True,
        # custom_drives=custom_drives,
    )
    # print(kitti_ds[0].cam_intrinsic)

    for f, frames in enumerate(kitti_ds.train_loader(batch_size=2)):
        frames = Frame.batch_list(frames)
        frames.get_view().render()
