from abc import ABC, abstractmethod
from typing import Any, List


class BaseLogger(ABC):
    """Base class for all loggers."""

    @abstractmethod
    def log_scalar(self, step: int, key: str, obj: Any):
        """Log a scalar to the current logger

        Parameters
        ----------
        step: int
            The current step
        key: str
            The key of the scalar
        obj: Any
            The scalar to log
        """
        pass

    @abstractmethod
    def log_image(self, step: int, key: str, image: Any):
        """Log an image to the current logger

        Parameters
        ----------
        step: int
            The current step
        key: str
            The key of the image
        image: Any
            The image to log
        """
        pass

    @abstractmethod
    def log_scatter(self, step: int, key: str, values: Any, column_names: List[str]):
        """Log a scatter plot to the current logger

        Parameters
        ----------
        step: int
            The current step
        key: str
            The key of the scatter plot
        values: Any
            The values of the scatter plot
        column_names: List[str]
            The names of the columns
        """
        pass

    @abstractmethod
    def log_hist(self, step: int, key: str, hist: Any):
        """Log a histogram to the current logger

        Parameters
        ----------
        step: int
            The current step
        key: str
            The key of the histogram
        hist: Any
            The histogram to log
        """
        pass
