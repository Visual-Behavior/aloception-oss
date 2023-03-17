import pytorch_lightning as pl
import alonet
import torch
from torch.utils.data.sampler import RandomSampler, SequentialSampler


class BaseDataModule(pl.LightningDataModule):
    """
    Base class for all data modules.
    """

    def __init__(
        self, args, **kwargs,
    ):
        super().__init__()
        alonet.common.pl_helpers.params_update(self, args, kwargs)

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("BaseDataModule")
        parser.add_argument("--batch_size", type=int, default=5, help="Batch size (Default: %(default)s)")
        parser.add_argument(
            "--num_workers", type=int, default=8, help="num_workers to use on the dataset (Default: %(default)s)"
        )
        parser.add_argument("--sequential_sampler", action="store_true", help="sample data sequentially (no shuffle)")
        parser.add_argument(
            "--sample", action="store_true", help="Download a sample for train/val process (Default: %(default)s)"
        )
        parser.add_argument("--train_on_val", action="store_true", help="Train on validation set (Default: %(default)s)")

        parser.add_argument("--no_aug", action="store_true", help="Disable data augmentation (Default: %(default)s)")
        return parent_parser

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

    @property
    def test_dataset(self):
        if not hasattr(self, "_test_dataset"):
            self.setup()
        return self._test_dataset

    @test_dataset.setter
    def test_dataset(self, new_dataset):
        self._test_dataset = new_dataset

    def train_transform(self, frames, **kwargs):
        """
        A structure to select the train transform function.
        Parameters
        ----------
        frames : aloscene.Frame
            Input frames
        Returns
        -------
        aloscene.Frame
        """
        if self.no_aug:
            return self._train_transform_no_aug(frames)
        else:
            return self._train_transform_aug(frames, **kwargs)

    def _train_transform_no_aug(self, frames):
        """
        Train_transform with no data augmentation.
        Parameters
        ----------
        frames : aloscene.Frame
            Input frames
        Returns
        -------
        aloscene.Frame
        """

        raise NotImplementedError("Should be implemented in child class.")

    def _train_transform_aug(self, frames):
        """
        Train_transform with data augmentation.
        Parameters
        ----------
        frames : aloscene.Frame
            Input frames
        Returns
        -------
        aloscene.Frame
        """

        raise NotImplementedError("Should be implemented in child class.")

    def val_transform(self, frames, **kwargs):
        """
        Val transform.
        Parameters
        ----------
        frames : aloscene.Frame
            Input frames
        Returns
        -------
        aloscene.Frame
        """

        raise NotImplementedError("Should be implemented in child class.")

    def setup(self, stage=None):
        """:attr:`train_dataset`, :attr:`val_dataset`, attr:`test_dataset` datasets setup
        Parameters
        ----------
        stage : str, optional
            Stage either `fit`, `validate`, `test` or `predict`, by default None"""

        raise NotImplementedError("Should be implemented in child class.")

    def train_dataloader(self, sampler: torch.utils.data = None):
        """Get train dataloader
        Parameters
        ----------
        sampler : torch.utils.data, optional
            Sampler to load batches, by default None
        Returns
        -------
        torch.utils.data.DataLoader
            Dataloader for training process
        """
        if sampler is None:
            sampler = RandomSampler if not self.sequential_sampler else SequentialSampler

        return self.train_dataset.train_loader(batch_size=self.batch_size, num_workers=self.num_workers, sampler=sampler)

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
        if sampler is None:
            sampler = SequentialSampler

        return self.val_dataset.train_loader(batch_size=self.batch_size, num_workers=self.num_workers, sampler=sampler)

    def test_dataloader(self, sampler: torch.utils.data = None):
        """Get test dataloader
        Parameters
        ----------
        sampler : torch.utils.data, optional
            Sampler to load batches, by default None
        Returns
        -------
        torch.utils.data.DataLoader
            Dataloader for inference process
        """
        if sampler is None:
            sampler = SequentialSampler

        return self.test_dataset.train_loader(batch_size=self.batch_size, num_workers=self.num_workers, sampler=sampler)
