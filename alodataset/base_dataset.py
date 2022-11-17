import os
import pickle
import requests
import shutil
import torch
import json
from tqdm import tqdm
from typing import List, Callable, Dict
from enum import Enum

from aloscene.io.utils.errors import InvalidSampleError
from aloscene import Frame
import aloscene

DATASETS_DOWNLOAD_PATHS = {
    "coco": "https://storage.googleapis.com/visualbehavior-sample/coco.pkl",
    "waymo": "https://storage.googleapis.com/visualbehavior-sample/waymo.pkl",
    "mot17": "https://storage.googleapis.com/visualbehavior-sample/mot17.pkl",
    "chairsSDHom": "https://storage.googleapis.com/visualbehavior-sample/chairsSDHom.pkl",
    "crowdhuman": "https://storage.googleapis.com/visualbehavior-sample/crowdhuman.pkl",
    "FlyingChairs2": "https://storage.googleapis.com/visualbehavior-sample/FlyingChairs2.pkl",
    "FlyingThings3DSubset": "https://storage.googleapis.com/visualbehavior-sample/FlyingThings3DSubset.pkl",
    "SintelDisparity": "https://storage.googleapis.com/visualbehavior-sample/SintelDisparity.pkl",
    "SintelFlow": "https://storage.googleapis.com/visualbehavior-sample/SintelFlow.pkl",
    "SintelMulti": "https://storage.googleapis.com/visualbehavior-sample/SintelMulti.pkl",
}


class Split(Enum):
    TRAIN: str = "train"
    VAL: str = "val"
    TEST: str = "test"

    Type: List[str] = ["train", "val", "test"]


def stream_loader(dataset, num_workers=2):
    """Get a stream loader from the dataset. Compared to the :func:`train_loader`
    the :func:`stream_loader` do not have batch dimension and do not shuffle the dataset.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Dataset to make dataloader
    num_workers : int
        Number of workers, by default 2

    Returns
    -------
    torch.utils.data.DataLoader
        A generator
    """
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=None, collate_fn=lambda d: dataset._collate_fn(d), num_workers=num_workers
    )
    return data_loader


def train_loader(dataset, batch_size=1, num_workers=2, sampler=torch.utils.data.RandomSampler, sampler_kwargs={}):
    """Get training loader from the dataset

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Dataset to make dataloader
    batch_size : int, optional
        Batch size, by default 1
    num_workers : int, optional
        Number of workers, by default 2
    sampler : torch.utils.data, optional
        Callback to sampler the dataset, by default torch.utils.data.RandomSampler

    Returns
    -------
    torch.utils.data.DataLoader
        A generator
    """
    if sampler is not None and not(isinstance(sampler, torch.utils.data.Sampler)):
        sampler = sampler(dataset, **sampler_kwargs)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        # batch_sampler=batch_sampler,
        sampler=sampler,
        batch_size=batch_size,
        drop_last=True,
        collate_fn=dataset._collate_fn,
        num_workers=num_workers,
    )

    return data_loader


def rename_data_to_none(data):
    """
    Temporarily remove data names until next call to `names` property.
    Necessary for pytorch operations that don't support named tensors
    """
    if isinstance(data, dict):
        for key in data:
            if isinstance(data[key], Frame):
                data[key].rename_(None, auto_restore_names=True)
    else:
        data.rename_(None, auto_restore_names=True)
    return data


def _user_prompt(message):
    res = input(message + "\033[93m")
    print("\033[0m", end="")  # Color reset
    return res


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        name,
        transform_fn: Callable = None,
        ignore_errors=False,
        print_errors=True,
        max_retry_on_error=3,
        retry_offset=20,
        sample: bool = False,
        **kwargs,
    ):
        """ Streaming dataset

        Parameters
        ----------
        name: str
            Name of this dataset. Usefull to automaticly set the `dataset_dir` property based
            on the dataset name.
        transform_fn: function
            transformation applied to each sample
        ignore_error : bool
            if True, when an invalid sample (corrupted data) is encountered while loading,
            another index is tried. More precisely : invalid sample at index `idx`
            is replaced with sample at index `idx + retry_offset % len(dataset)`.
            Defaults to False.
        print_error : bool
            if True, when an invalid sample is ignored, a warning message is printed.
            Defaults to True.
        max_retry_on_error: int
            if ignore_error is true, replacing an invalid samples by a sampling at a different index
            is not guaranteed to succeed, because the sample at the new index can also be invalid.
            In this case, we can retry with new index again, at most `max_retry_on_error` times.
            If all tried samples are invalid, an InvalidSampleError is raised.
        retry_offset : int
            see description of `ignore_errors` parameter.
        sample : bool
            Download (or not) a dataset sample from internet and replace the default dataset_dir,
            by default False.
        """

        super(BaseDataset, self).__init__(**kwargs)
        self.name = name
        self.sample = sample
        self.dataset_dir = self.get_dataset_dir()
        if self.sample:
            self.items = self.download_sample()
        else:
            self.items = []
        self.transform_fn = transform_fn
        self.ignore_errors = ignore_errors
        self.print_errors = print_errors
        self.retry_offset = retry_offset
        self.max_retry_on_error = max_retry_on_error

    def getitem(self):
        raise Exception("Not implemented Error")

    def __repr__(self) -> str:
        """Print class format inspired by VisionDataset"""
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(len(self))]
        if self.dataset_dir is not None:
            body.append("Root location: {}".format(self.dataset_dir))
        body += "".splitlines()
        if hasattr(self, "transform_fn") and self.transform_fn is not None:
            body += [repr(self.transform_fn)]
        lines = [head] + [" " * 4 + line for line in body]
        return "\n".join(lines)

    def getitem_ignore_errors(self, idx):
        """
        Try to get item at index idx.
        If data is invalid, retry at a shifted index.
        Repeat until the max limit of authorized tries is reached.
        """
        nb_retry = 0
        while nb_retry <= self.max_retry_on_error:
            try:
                data = self.getitem(idx)
                return data
            except InvalidSampleError as e:
                if self.print_errors:
                    print(f"\n{e}\n")
                nb_retry += 1
                idx = (idx + self.retry_offset) % len(self)
        # max limit reached
        max_try = self.max_retry_on_error
        raise InvalidSampleError(f"Reached the limit of {max_try} consecutive corrupted samples.")

    def __getitem__(self, idx):
        if self.sample:
            data = self.items[idx]
        else:
            data = self.getitem_ignore_errors(idx) if self.ignore_errors else self.getitem(idx)
        if self.transform_fn is not None:
            data = self.transform_fn(data)

        # Rename datas to None before to return the result
        # (Name support not yet supported in the datapipeline)
        data = rename_data_to_none(data)
        return data

    def get(self, idx: int) -> Dict[str, aloscene.Frame]:
        """Get a specific element in the dataset. Note that usually
        we could call directly dataset[idx] instead of dataset.get(idx). But right now
        __getitem__ is ony fully support through the `stream_loader` and the `train_loader`.

        Parameters
        ----------
        idx: int
            Index of the element to get
        """
        data = self.getitem(idx)
        if self.transform_fn is not None:
            data = self.transform_fn(data)
        return data

    def get_dataset_dir(self) -> str:
        """Look for dataset_dir based on the given name. To work properly a alodataset_config.json
        file must be save into /home/USER/.aloception/alodataset_config.json
        """
        if self.sample:
            return os.path.join(self.vb_folder, "samples")

        streaming_dt_config = os.path.join(self.vb_folder, "alodataset_config.json")
        if not os.path.exists(streaming_dt_config):
            self.set_dataset_dir(None)
        with open(streaming_dt_config) as f:
            content = json.loads(f.read())

        # Diferent cases
        if self.name not in content:
            dataset_dir = self.set_dataset_dir(None)
        elif not os.path.exists(content[self.name]):
            dataset_dir = self.set_dataset_dir(content[self.name])
            # raise Exception(f"{self.name} not added into streaming_dt_config")
        else:
            dataset_dir = content[self.name]
        return dataset_dir

    def set_dataset_dir(self, dataset_dir: str):
        """Set the dataset_dir into the config file. This method will
        write the  path into /home/USER/.aloception/alodataset_config.json by replacing the current one
        (if any)

        Parameters
        ----------
        dataset_dir: str
            Path to the new directory
        """
        streaming_dt_config = os.path.join(self.vb_folder, "alodataset_config.json")
        if not os.path.exists(streaming_dt_config):
            with open(streaming_dt_config, "w") as f:  # Json init as empty config
                json.dump(dict(), f, indent=4)
        with open(streaming_dt_config) as f:
            content = json.loads(f.read())

        if dataset_dir is None:
            if self.name in DATASETS_DOWNLOAD_PATHS:
                dataset_dir = _user_prompt(
                    f"{self.name} does not exist in config file. "
                    + "Do you want to download and use a sample?: (Y)es or (N)o: "
                )
                if dataset_dir.lower() in ["y", "yes"]:  # Download sample and change root directory
                    self.sample = True
                    return os.path.join(self.vb_folder, "samples")
            dataset_dir = _user_prompt(f"Please write a new root directory for {self.name} dataset: ")
            dataset_dir = os.path.expanduser(dataset_dir)

        # Save the config
        if not os.path.exists(dataset_dir):
            dataset_dir = _user_prompt(
                f"[WARNING] {dataset_dir} path does not exists for dataset: {self.name}. "
                + "Please write a new directory:"
            )
            dataset_dir = os.path.expanduser(dataset_dir)
            if not os.path.exists(dataset_dir):
                raise Exception(f"{dataset_dir} path does not exists for dataset: {self.name}")

        content[self.name] = dataset_dir
        with open(streaming_dt_config, "w") as f:  # Save new directory
            json.dump(content, f, indent=4)

        # Set the new dataset_dir on the class
        self.dataset_dir = dataset_dir

        return dataset_dir

    def __len__(self):
        return len(self.items)

    @property
    def vb_folder(self):
        home = os.getenv("HOME")
        alofolder = os.path.join(home, ".aloception")
        if not os.path.exists(alofolder):  # Folder creates if doesnt exist
            os.mkdir(alofolder)
        return alofolder

    def _collate_fn(self, batch_data):
        """Streamer collat fn"""
        return batch_data

    def stream_loader(self, num_workers=2):
        """Get a stream loader from the dataset. Compared to the :func:`train_loader`
        the :func:`stream_loader` do not have batch dimension and do not shuffle the dataset.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            Dataset to make dataloader
        num_workers : int
            Number of workers, by default 2

        Returns
        -------
        torch.utils.data.DataLoader
            A generator
        """
        return stream_loader(self, num_workers=num_workers)

    def train_loader(self, batch_size=1, num_workers=2, sampler=torch.utils.data.RandomSampler, sampler_kwargs={}):
        """Get training loader from the dataset

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            Dataset to make dataloader
        batch_size : int, optional
            Batch size, by default 1
        num_workers : int, optional
            Number of workers, by default 2
        sampler : torch.utils.data, optional
            Callback to sampler the dataset, by default torch.utils.data.RandomSampler

        Returns
        -------
        torch.utils.data.DataLoader
            A generator
        """
        return train_loader(self, batch_size=batch_size, num_workers=num_workers, sampler=sampler, sampler_kwargs=sampler_kwargs    )

    def prepare(self):
        """Prepare the dataset. Not all child class need to implement this method.
        However, for some classes, it could be effective to prepare the dataset either
        to be faster later or to reduce the storage of the whole dataset.
        """
        pass

    def download_sample(self) -> str:
        """Download a dataset sample, replacing the original dataset.

        Returns
        -------
        str
            New directory: self.vb_folder+"samples"

        Raises
        ------
        Exception
            The dataset must be one of DATASETS_DOWNLOAD_PATHS
        """
        streaming_sample = os.path.join(self.vb_folder, "samples")

        if self.name not in DATASETS_DOWNLOAD_PATHS:
            raise Exception(f"Impossible to download {self.name} sample.")

        if not os.path.exists(streaming_sample):
            os.makedirs(streaming_sample)

        src = DATASETS_DOWNLOAD_PATHS[self.name]
        dest = os.path.join(streaming_sample, self.name + ".pkl")
        if not os.path.exists(os.path.join(dest)):
            print(f"Download {self.name} sample...")
            if "http" in src:
                with open(dest, "wb") as f:
                    response = requests.get(src, stream=True)
                    total_length = response.headers.get("content-length")

                    if total_length is None:  # no content length header
                        f.write(response.content)
                    else:
                        pbar = tqdm()
                        pbar.reset(total=int(total_length))  # initialise with new `total`
                        for data in response.iter_content(chunk_size=4096):
                            f.write(data)
                            pbar.update(len(data))
            else:
                shutil.copy2(src, dest)

        with open(dest, "rb") as f:
            sample = pickle.load(f)

        return sample
