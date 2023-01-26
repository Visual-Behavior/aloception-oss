import pytorch_lightning as pl
import alonet
import torch


class BaseDataModule(pl.LightningDataModule):
    """
    Here be description
    """

    def __init__(
        self, args, **kwargs,
    ):
        """
        Here be description
        """

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

    def train_transform(self, frames, **kwargs):
        """
        Here be description"""
        if self.no_aug:
            return self._train_transform_no_aug(frames)
        else:
            return self._train_transform_aug(frames, **kwargs)

    def _train_transform_no_aug(self, frames):
        """
        Here be description
        """

        raise NotImplementedError("Should be implemented in child class.")

    def _train_transform_aug(self, frames):
        """
        Here be description
        """

        raise NotImplementedError("Should be implemented in child class.")

    def val_transform(self, frames, **kwargs):
        """
        Here be description
        """

        raise NotImplementedError("Should be implemented in child class.")

    def _setup_train_dataset(self):
        self._train_dataset = TartanairDataset(
            sequences=self.process_sequences(self.train_sequences, self.difficulty),
            cameras=["left"],
            labels=["flow"],
            pose_format="camera",
            sequence_size=2,
            transform_fn=self.train_val_transform,
        )

    def _setup_val_dataset(self):
        self._val_dataset = TartanairDataset(
            sequences=self.process_sequences(self.val_sequences, self.difficulty),
            cameras=["left"],
            labels=["flow"],
            pose_format="camera",
            sequence_size=2,
            transform_fn=self.train_val_transform,
        )

    def _setup_test_dataset(self):
        self._test_dataset = TartanairDataset(
            sequences=self.process_sequences(self.test_sequences, self.difficulty),
            cameras=["left"],
            labels=["flow"],
            pose_format="camera",
            sequence_size=2,
            transform_fn=self.test_predict_transform,
        )

    def setup(self, stage=None):
        self._setup_train_dataset()
        self._setup_val_dataset()
        self._setup_test_dataset()

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
