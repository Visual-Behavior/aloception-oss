from abc import ABC, abstractmethod
import warnings
from torch.utils.data import DataLoader


class BaseDataModule(ABC):
    """
    Base class for all data modules.
    """

    def __init__(self, batch_size: int, num_workers: int = 8):
        """
        Initialize the data module

        Parameters
        ----------
        batch_size : int
            Batch size for the dataloader
        num_workers : int
            Number of workers for the dataloader
        """
        super().__init__()
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._train_dataloader = None
        self._val_dataloader = None
        self._test_dataloader = None

    @property
    def batch_size(self) -> int:
        """
        Batch size for the dataloader
        """
        return self._batch_size

    @property
    def num_workers(self) -> int:
        """
        Number of workers for the dataloader
        """
        return self._num_workers

    @property
    def train_dataloader(self) -> DataLoader:
        """
        Train dataloader
        """
        return self._train_dataloader

    @property
    def val_dataloader(self) -> DataLoader:
        """
        Validation dataloader
        """
        return self._val_dataloader

    @property
    def test_dataloader(self) -> DataLoader:
        """
        Test dataloader
        """
        return self._test_dataloader

    @batch_size.setter
    def batch_size(self, value: int):
        """
        Set batch size

        Parameters
        ----------
        value : int
            Batch size
        """
        self._batch_size = value

    @num_workers.setter
    def num_workers(self, value: int):
        """
        Set number of workers

        Parameters
        ----------
        value : int
            Number of workers
        """
        self._num_workers = value

    @train_dataloader.setter
    def train_dataloader(self, value: DataLoader):
        """
        Set train dataloader

        Parameters
        ----------
        value : DataLoader
            Train dataloader
        """
        self._train_dataloader = value

    @val_dataloader.setter
    def val_dataloader(self, value: DataLoader):
        """
        Set validation dataloader

        Parameters
        ----------
        value : DataLoader
            Validation dataloader
        """
        self._val_dataloader = value

    def setup(self, stage: str):
        """
        :attr:`train_dataloader`, :attr:`val_dataloader`, attr:`test_dataloader` dataloaders setup
        Parameters
        ----------
        stage : str, optional
            Stage either `training`, `validation`, `testing`, by default None
        """
        assert stage in [
            "training",
            "validation",
            "testing",
        ], "Stage must be one of: training, validation, testing"

        if stage == "training":
            self.train_dataloader = self.setup_train_dataloader()
            self.val_dataloader = self.setup_val_dataloader()
        elif stage == "validation":
            self.val_dataloader = self.setup_val_dataloader()
        elif stage == "testing":
            self.test_dataloader = self.setup_test_dataloader()

    @abstractmethod
    def setup_train_dataloader(self) -> DataLoader:
        """
        Setup train dataloader
        """
        pass

    @abstractmethod
    def setup_val_dataloader(self) -> DataLoader:
        """
        Setup val dataloader
        """
        pass

    def setup_test_dataloader(self) -> DataLoader:
        """
        Setup test dataloader
        """
        warnings.warn("Test dataloader is not implemented in the base class.")
        pass
