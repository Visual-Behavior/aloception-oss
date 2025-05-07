import os
from typing import Any, List
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np

from .base_logger import BaseLogger
from alonet.common.helpers import only_main_rank


class TensorboardLogger(BaseLogger):
    """TensorBoard logger."""

    def __init__(
        self,
        name: str,
        project: str,
        save_dir: str = ".",
        resume: bool = False,
    ):
        """Initialize TensorBoard logger.

        Parameters
        ----------
        name : str
            Name of the run
        project : str
            Name of the project
        save_dir : str, optional
            Directory to save TensorBoard logs, by default "."
        resume : bool, optional
            Whether to resume logging from a previous run, by default False
        """
        self.name = name
        self.project = project
        self.save_dir = save_dir if save_dir is not None else os.path.join("tensorboard", self.project, self.name)
        self.resume = resume

        # Create the log directory
        log_dir = os.path.join(self.save_dir, self.project, self.name)
        os.makedirs(log_dir, exist_ok=True)

        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=log_dir, purge_step=None if resume else 0)

    @only_main_rank
    def log_scalar(self, step: int, key: str, obj: Any):
        """Log a scalar to TensorBoard.

        Parameters
        ----------
        step : int
            The current step
        key : str
            Tag name of the log
        obj : Any
            The scalar to log
        """
        self.writer.add_scalar(key, obj, step)

    @only_main_rank
    def log_image(self, step: int, key: str, image: Any):
        """Log an image to TensorBoard.

        Parameters
        ----------
        step : int
            The current step
        key : str
            Tag name of the log
        image : Any
            The image to log (should be a tensor, numpy array, or PIL image).
            The image must be in the shape of (H, W, C).
        """
        self.writer.add_image(key, image, step, dataformats="HWC")

    @only_main_rank
    def log_scatter(self, step: int, key: str, values: Any, column_names: List[str]):
        """Log a scatter plot to TensorBoard.

        Parameters
        ----------
        step : int
            The current step
        key : str
            Tag name of the log
        values : torch.Tensor
            Values of the graph tensor (N, 2)
        column_names : List[str]
            Names of axes (x axis, y axis)
        """
        if isinstance(values, torch.Tensor):
            values = values.cpu().numpy()
        elif not isinstance(values, np.ndarray):
            values = np.array(values)

        # Create figure using matplotlib
        import matplotlib.pyplot as plt

        fig = plt.figure()
        plt.scatter(values[:, 0], values[:, 1])
        plt.xlabel(column_names[0])
        plt.ylabel(column_names[1])
        plt.title(key)

        # Log the figure
        self.writer.add_figure(key, fig, step)
        plt.close(fig)

    @only_main_rank
    def log_hist(self, step: int, key: str, hist: Any):
        """Log a histogram to TensorBoard.

        Parameters
        ----------
        step : int
            The current step
        key : str
            Tag name of the log
        hist : torch.Tensor
            Tensor containing the values to create histogram
        """
        self.writer.add_histogram(key, hist, step)

    def __del__(self):
        """Cleanup: close the TensorBoard writer when the logger is destroyed."""
        if hasattr(self, "writer"):
            self.writer.close()
