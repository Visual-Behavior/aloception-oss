from abc import ABC, abstractmethod


class BaseLogger(ABC):
    """Base class for all loggers."""

    @abstractmethod
    def log_scalar(self, *args, **kwargs):
        pass

    @abstractmethod
    def log_image(self, *args, **kwargs):
        pass

    @abstractmethod
    def log_scatter(self, *args, **kwargs):
        pass

    @abstractmethod
    def log_hist(self, *args, **kwargs):
        pass
