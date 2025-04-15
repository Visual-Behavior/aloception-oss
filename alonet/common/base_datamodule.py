from abc import ABC, abstractmethod
import warnings
from torch.utils.data import DataLoader


class BaseDataModule(ABC):
    """
    Base class for all data modules.
    """

    def __init__(self, batch_size: int, num_workers: int = 8):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None

    def setup(self, stage: str):
        """
        :attr:`train_dataloader`, :attr:`val_dataloader`, attr:`test_dataloader` dataloaders setup
        Parameters
        ----------
        stage : str, optional
            Stage either `fit`, `validate`, `test` or `predict`, by default None
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
