"""LightningDataModule base to use dataset in Detr trainer models.

See Also
--------
`LightningDataModule <https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html>`_ for more
information about to create data modules from
`Dataset and Dataloader <https://pytorch.org/tutorials/beginner/basics/data_tutorial.html>`_
"""

import torch
from argparse import Namespace, ArgumentParser, _ArgumentGroup
from typing import Optional

from alodataset import transforms as T  # , Split
import pytorch_lightning as pl

import alonet
import aloscene


class Data2Detr(pl.LightningDataModule):
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

    def __init__(self, args: Namespace = None, **kwargs):
        # Update class attributes with args and kwargs inputs
        super().__init__()
        alonet.common.pl_helpers.params_update(self, args, kwargs)

        self.size = list(self.size)
        if len(self.size) == 1:
            self.size[0] = self.size[0] if self.size[0] is None else int(self.size[0])
            self.size = (self.size[0], self.size[0])
        elif len(self.size) == 2:
            self.size[0] = self.size[0] if self.size[0] is None else int(self.size[0])
            self.size[1] = self.size[1] if self.size[1] is None else int(self.size[1])
        else:
            raise Exception("Must be provided one or two elements in size argument.")

        self.args = args

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

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser, parser: _ArgumentGroup = None):
        """Append the respect arguments to parser object

        Parameters
        ----------
        parent_parser : ArgumentParser
            Object with previous arguments to append
        parser : ArgumentParser._ArgumentGroup, optional
            Argument group to append the parameters, by default None

        Returns
        -------
        ArgumentParser
            Arguments updated
        """
        parser = parent_parser.add_argument_group("DataModule") if parser is None else parser
        parser.add_argument("--batch_size", type=int, default=2, help="Batch size to use (default %(default)s)")
        parser.add_argument("--train_on_val", action="store_true")
        parser.add_argument(
            "--num_workers",
            type=int,
            default=8,
            help="num_workers to use on the CocoBaseDataset (default %(default)s)",
        )
        parser.add_argument(
            "--no_augmentation", action="store_true", help="Do not use augmentation to train the model"
        )
        parser.add_argument(
            "--size",
            type=int,
            default=(None, None),
            nargs="+",
            help="If no augmentation (--no_augmentation) is used, --size can be used to resize all the frame.",
        )
        parser.add_argument(
            "--sample", action="store_true", help="Download a sample for train/val process (Default: %(default)s)"
        )
        return parent_parser

    def train_transform(self, frame: aloscene.Frame, same_on_sequence: bool = True, same_on_frames: bool = False):
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
        self, frame: aloscene.Frame, same_on_sequence: bool = True, same_on_frames: bool = False,
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
            [800], max_size=1333, same_on_sequence=same_on_sequence, same_on_frames=same_on_frames
        )(frame)

        return frame.norm_resnet()

    def setup(self, stage: Optional[str] = None):
        """:attr:`train_dataset` and :attr:`val_dataset` datasets setup, follow the parameters used
        in class declaration.

        Parameters
        ----------
        stage : str, optional
            Stage either `fit`, `validate`, `test` or `predict`, by default None
        """
        raise Exception("This class must be inhert and set ``train_dataset`` and ``val_dataset`` class attributes")

    def train_dataloader(self):
        """Get train dataloader

        Returns
        -------
        torch.utils.data.DataLoader
            Dataloader for training process
        """
        return self.train_dataset.train_loader(batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self, sampler: torch.utils.data = None):
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
        return self.val_dataset.train_loader(batch_size=self.batch_size, num_workers=self.num_workers, sampler=sampler)
