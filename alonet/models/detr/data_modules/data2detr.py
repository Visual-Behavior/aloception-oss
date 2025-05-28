from abc import abstractmethod
import warnings
import torch
from torch.utils.data import Dataset, DataLoader

from alodataset import transforms as T

from alonet.common import BaseDataModule
import aloscene
from torch.utils.data.sampler import RandomSampler, SequentialSampler


class Data2Detr(BaseDataModule):
    """
    Parameters
    ----------
    batch_size : int, optional
        Batch size to use
    train_on_val : bool, optional
        Use train on validation
    num_workers : int, optional
        Nummer of workers to use
    no_augmentation : bool, optional
        Do not use augmentation to train the model
    size : tuple, optional
        If no augmentation (--no_augmentation) is used, --size can be used to resize all the frame.
    sample : bool, optional
        Use Sample instead of all dataset, by default False
    args : Namespace, optional
        Attributes stored in specific Namespace, by default None

    Raises
    ------
    Exception
        Size argument must be one or two elements

    Notes
    -----
    Arguments entered by the user (kwargs) will replace those stored in args attribute
    """

    def __init__(
        self,
        no_augmentation: bool = False,
        size: tuple = (None, None),
        sample: bool = False,
        sequential: bool = False,
        **kwargs
    ):
        """
        Data module to use for training and validation of DETR model.
        Parameters
        ----------
        no_augmentation : bool, optional
            Do not use augmentation to train the model
        size : tuple, optional
            If no augmentation (--no_augmentation) is used, --size can be used to resize all the frame.
        sample : bool, optional
        """
        # Update class attributes with args and kwargs inputs
        super().__init__(**kwargs)

        self.size = list(size)
        if len(self.size) == 1:
            self.size[0] = self.size[0] if self.size[0] is None else int(self.size[0])
            self.size = (self.size[0], self.size[0])
        elif len(self.size) == 2:
            self.size[0] = self.size[0] if self.size[0] is None else int(self.size[0])
            self.size[1] = self.size[1] if self.size[1] is None else int(self.size[1])
        else:
            raise Exception("Must be provided one or two elements in size argument.")

        self.no_augmentation = no_augmentation
        self.sample = sample
        self.sequential = sequential

    @property
    def train_dataset(self):
        if not hasattr(self, "_train_dataset"):
            self.setup()
        return self._train_dataset

    @train_dataset.setter
    def train_dataset(self, new_dataset):
        self._train_dataset = new_dataset

    @property
    def val_dataset(self):
        if not hasattr(self, "_val_dataset"):
            self.setup()
        return self._val_dataset

    @val_dataset.setter
    def val_dataset(self, new_dataset):
        self._val_dataset = new_dataset

    def train_transform(
        self,
        frame: aloscene.Frame,
        same_on_sequence: bool = True,
        same_on_frames: bool = False,
    ):
        """Transorm requered to train on each frame

        Parameters
        ----------
        frame : :mod:`~aloscene.frame`
            Input frame to transform
        same_on_sequence : bool, optional
            Use same data augmentation size of each sequence, by default True
        same_on_frames : bool, optional
            Use same data augmentation size of each frame, by default False

        Returns
        -------
        :mod:`~aloscene.frame`
            Frame with respective changes by transform function
        """
        if self.no_augmentation:
            if self.size[0] is not None and self.size[1] is not None:
                frame = T.Resize((self.size[0], self.size[1]))(frame)
            return frame.norm_resnet()

        scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
        frame = T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.RandomSelect(
                    T.RandomResizeWithAspectRatio(scales, max_size=1333),
                    T.Compose(
                        [
                            T.RandomResizeWithAspectRatio([400, 500, 600]),
                            T.RandomSizeCrop(384, 600),
                            T.RandomResizeWithAspectRatio(scales, max_size=1333),
                        ]
                    ),
                ),
            ],
            same_on_sequence=same_on_sequence,
            same_on_frames=same_on_frames,
        )(frame)

        return frame.norm_resnet()

    def val_transform(
        self,
        frame: aloscene.Frame,
        same_on_sequence: bool = True,
        same_on_frames: bool = False,
    ):
        """Transform requered to valid on each frame

        Parameters
        ----------
        frame : :mod:`~aloscene.frame`
            Input frame to transform
        same_on_sequence : bool, optional
            Use same data augmentation size of each sequence, by default True
        same_on_frames : bool, optional
            Use same data augmentation size of each frame, by default False

        Returns
        -------
        :mod:`~aloscene.frame`
            Frame with respective changes by transform function
        """
        if self.no_augmentation:
            if self.size[0] is not None and self.size[1] is not None:
                frame = T.Resize((self.size[1], self.size[1]))(frame)
            return frame.norm_resnet()

        # Reszie keeping aspect ratio
        frame = T.RandomResizeWithAspectRatio(
            [800],
            max_size=1333,
            same_on_sequence=same_on_sequence,
            same_on_frames=same_on_frames,
        )(frame)

        return frame.norm_resnet()

    @abstractmethod
    def setup_train_dataset(self) -> Dataset:
        """Get train dataset

        Returns
        -------
        torch.utils.data.Dataset
        """
        warnings.warn("Must be implemented in child class")

    @abstractmethod
    def setup_val_dataset(self) -> Dataset:
        """Get val dataset

        Returns
        -------
        torch.utils.data.Dataset
        """
        warnings.warn("Must be implemented in child class")

    def setup_train_dataloader(self) -> DataLoader:
        """Get train dataloader

        Returns
        -------
        torch.utils.data.DataLoader
            Dataloader for training process
        """
        self.train_dataset = self.setup_train_dataset()
        return self.train_dataset.train_loader(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=SequentialSampler if self.sequential else RandomSampler,
        )

    def setup_val_dataloader(self, sampler: torch.utils.data = None) -> DataLoader:
        """Get val dataloader

        Parameters
        ----------
        sampler : torch.utils.data, optional
            Sampler to load batches, by default None

        Returns
        -------
        torch.utils.data.DataLoader
            Dataloader for validation process
        """
        self.val_dataset = self.setup_val_dataset()
        return self.val_dataset.train_loader(batch_size=self.batch_size, num_workers=self.num_workers, sampler=sampler)
