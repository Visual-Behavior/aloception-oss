from re import L
import aloscene
from alodataset import BaseDataset

import os
import cv2
import glob
import numpy as np
from typing import List, Union, Dict


class FromDirectoryDataset(BaseDataset):
    """Data iterator from directory 
    
    Parameters
    ----------
        dirs : List[str]
            List of directories paths to load images from. Default None.
        
        interval : List[flaot]
            Use for restrciting the number of samples. Default [0, 1].
                Example interval = [0.2, 0.4] will only load samples from 20%th to 40%th.

    Raises
    ------
        Exception
            All directories are empty.

        Exception
            One of the directories does not exist.

        Exception
            One of the paths is not a directory.

    Examples
    --------
        >>> # from list of paths.
        >>> path1 = "/PATH/TO/DATA/DIR1"
        >>> path2 = "/PATH/TO/DATA/DIR2"

        >>> data = FromDirectoryDataset(dirs=[path1, path2])
        >>> for i in range(len(data)):
        >>>    print(data[i].shape)

        >>> # from dict of list of paths.
        >>> path0 = "/PATH/TO/DATA/DIR0"
        >>> path1 = "/PATH/TO/DATA/DIR1"
        >>> path2 = "/PATH/TO/DATA/DIR2"
        >>> path3 = "/PATH/TO/DATA/DIR3"

        >>> data = FromDirectoryDataset(dirs={"key1": [path0, path1], "key2": [path2, path3]])
        >>> for i in range(len(data)):
        >>>    print(data[i]["key1"].shape)
        >>>    print(data[i]["key2"].shape)

    """
    def __init__(
            self,
            dirs : Union[List[str], Dict] = None,
            slice : list = [0, 1],
            name : str = "from_dir",
            **kwargs
            ):
        super().__init__(
            name,
            **kwargs
            )
        assert not self.sample, "Can not sample this dataset"
        assert dirs not in [None, [], {}], "List of directories not provided"
    
        assert len(slice) == 2, "Slice arg should list of 2 elements"
        assert slice[0] < slice[1], "Element at index 1 should be greater than elemnt at index 0."
        assert slice[0] >= 0 and slice[1] <=1, "Unvalid slice, values should be between 0 and 1"

        if isinstance(dirs , list):
            self.items = self._extract_dir_path(dirs)
            
        elif isinstance(dirs, dict):
            titems = []
            keys = list(dirs.keys())
            for key in keys:
                p_list = dirs[key]
                d_path = {key: self._extract_dir_path(p_list)}
                titems.append(d_path)
            # Hypotesis : all dirs have the same number of elements
            for i in range(len(titems[0][keys[0]])):
                sample = {}
                for key in keys:
                    sample[key] = titems[keys.index(key)][key][i]
                self.items.append(sample)

        # Slicing
        l_ = len(self.items)
        s_ = int(l_ * slice[0])
        e_ = int(l_ * slice[1])
        self.items = self.items[s_: e_]

        # Samples check.
        if not len(self.items):
            raise Exception("Empty dir, no .png .jpg files found")

    def _extract_dir_path(self, dirs):
        items = []
        for dir in dirs:
            if not os.path.exists(dir):
                raise Exception(f"Directory not found: {dir}")
            if not os.path.isdir(dir):
                raise Exception(f"{dir} is not a directory")
        
        for dir in dirs:
            gpath = os.path.join(dir, "*")
            files = sorted(glob.glob(gpath))
            files = list(filter(self._filter_img_path, files))
            items += files
        return items
    
    @staticmethod
    def _filter_img_path(path):
        ends = [".png", ".jpg"]
        return any([path.endswith(e) for e in ends])
    
    @staticmethod
    def _load_frame(path):
        if path.endswith((".png", ".jpg")):
            frame = cv2.imread(path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.moveaxis(frame, 2, 0)
            frame = aloscene.Frame(frame, names=tuple("CHW"))
            return frame
        else:
            raise Exception(f"Unknown extention for file {path}")

    def getitem(self, idx):
        item = self.items[idx]
        if isinstance(item, str):
            frame = self._load_frame(item)
        elif isinstance(item, dict):
            frame = {k: self._load_frame(path) for k, path in item.items()}
        else:
            raise Exception("Unknown item format")
        return frame
    
    def set_dataset_dir(self, dataset_dir: str):
        pass


if __name__ == "__main__":
    path1 = "amusement/amusement/Easy/P001/image_right"
    path2 = "amusement/amusement/Easy/P001/image_left"

    data = FromDirectoryDataset(dirs={"right": [path1, path1], "left": [path2, path2]}, slice=[0.2, 0.3])
    for i in range(len(data)):
        print(data[i])
