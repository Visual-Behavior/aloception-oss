from abc import ABC, abstractmethod

from alonet.common.helpers import is_main_rank, is_dist_avail_and_initialized


def rank_zero_only(func):
    def wrapper(*args, **kwargs):
        if (is_dist_avail_and_initialized() and is_main_rank()) or not is_dist_avail_and_initialized():
            return func(*args, **kwargs)
        else:
            return

    return wrapper


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
