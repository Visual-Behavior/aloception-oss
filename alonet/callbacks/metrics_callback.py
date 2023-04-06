from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
import torch
import lightning as pl
from typing import *
import aloscene
import wandb
from collections import deque
import numpy as np
import torch.distributed as dist
import alonet
from pytorch_lightning.utilities import rank_zero_only
from alonet.common.logger import log_hist, log_scatter


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


class MetricsCallback(pl.Callback):
    """Callback for any training that need to log metrics and
    loss results. To be used, the model outputs must be a dict
    containing the key "metrics". During training,
    the scalar values will be averaged over `step_smooth` steps.

    Additionally, if the metrics name start with histogram or scatter, this
    class will treat them as such and will try to log histogram/scatter on wandb.

    Parameters
    ----------
    step_smooth: int
        size in steps of the window for computing the moving average of scalar metrics .
    val_names : list[str]
        names associated with each val_dataloader.
        Stats will be computed separately for each dataset and logged in wandb with the associated val_name as prefix.

    Examples
    --------
    Here is an example of an expected model forward output.

    >>> outputs = {
            "metrics": {
                "loss": loss_value,
                "cross_entropy": cross_entropy_value,
                "histogram": torch.tensor(n,),
                "scatter": (["name_x_axis", "name_y_axis"], torch.tensor(n, 2)),
            }
        }
    """

    def __init__(self, step_smooth=100, val_names=None):
        """The callback load frames every `trainer.log_every_n_steps` training step as well as once
        every validation step on the given `val_frames`
        """
        self.val_names = val_names
        self.metrics = {}
        self.metrics_info = {}
        self.val_metrics = self._empty_val_metrics()
        self.val_metrics_info = self._empty_val_metrics()
        self.step_smooth = step_smooth
        super().__init__()

    @property
    def _n_val_datasets(self):
        if self.val_names is None:
            return 1
        return len(self.val_names)

    def _empty_val_metrics(self):
        return [{} for _ in range(self._n_val_datasets)]

    def on_train_epoch_end(self, trainer, pl_module):
        self.metrics = {}
        self.metrics_info = {}

    def on_validation_epoch_end(self, trainer, pl_module):
        self.val_metrics = self._empty_val_metrics()
        self.val_metrics_info = self._empty_val_metrics()

    def _process_train_metrics(self, outputs):
        """
        Process the metrics, w.r.t. to the type of variables.

        For histogram or scatter, data is stored for future use.
        For numerical metric, the rolling mean is updated

        Parameters
        ----------
        outputs: dict
            outputs of a training step
        """
        for key in outputs["metrics"]:
            n_value = outputs["metrics"][key]
            if isinstance(n_value, torch.Tensor):
                n_value = n_value.detach().cpu()
            if key not in self.metrics:
                if "histogram" in key:
                    self.metrics[key] = [n_value]
                elif "scatter" in key:
                    self.metrics[key] = [n_value[1]]
                    self.metrics_info[key] = n_value[0]
                else:
                    self.metrics[key] = deque([n_value], maxlen=self.step_smooth)
            else:
                if "histogram" in key:
                    self.metrics[key].append(n_value)
                elif "scatter" in key:
                    self.metrics[key].append(n_value[1])
                else:
                    self.metrics[key].append(float(n_value))

    def _log_train_metrics(self, pl_module, trainer):
        # Log the results
        for key in self.metrics:
            if "histogram" in key and len(self.metrics[key]) > 0:
                hist = torch.cat(self.metrics[key]).to("cpu")
                log_hist(trainer, f"train/{key}", hist)
                self.metrics[key] = []
            elif "scatter" in key and len(self.metrics[key]) > 0:
                values = []
                for v in self.metrics[key]:
                    values += v.to("cpu").numpy().tolist()
                log_scatter(trainer, f"train/{key}", values, self.metrics_info[key])
                self.metrics[key] = []
            elif "histogram" not in key and "scatter" not in key:
                pl_module.log(
                    f"train/{key}", np.mean(self.metrics[key]), on_step=True, on_epoch=False, rank_zero_only=True
                )

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Method called after each training batch. This class is a pytorch lightning callback, therefore
        this method will by automatically called by pytorch lightning.

        Parameters
        ----------
        trainer: pl.Trainer
            Pytorch lightning trainer
        pl_module: pl.LightningModule
            Pytorch lightning module
        outputs:
            Training/Validation step outputs of the pl.LightningModule class.
            The `metrics` key is expected for this callback to work properly.
            m_outputs[`metrics`] must be a dict.
            For each key,
            - if the key contains the keyword 'histogram'
            The value of the tensor will be aggregate to compute an histogram all `trainer.log_every_n_steps`.
            - If the key contains the keyword 'scatter'
            The value of the key must of a tuple, (scatter_names: tuple, torch.tensor) with the first element
            being a list of len 2 with the name of the X axis and the Y axis. The second element will be
            a tensor of size (N, 2).
            - Othwerwise
            The tensor is expected to be a single scaler with the value to log
        batch: list
            Batch coming from the dataloader. Usually, a list of frame.
        batch_idx: int
            Id of the batch
        dataloader_idx: int
            Id of the dataloader.
        """
        if trainer.logger is None:
            return
        if not isinstance(outputs, dict) or "metrics" not in outputs:
            raise Exception(
                "The lightning training_step must return a dict with the  `metrics` key to use this Callback"
            )

        self._process_train_metrics(outputs)
        should_accumulate = trainer.fit_loop._should_accumulate()
        if should_accumulate or (trainer.global_step + 1) % trainer.log_every_n_steps != 0:
            return

        self._log_train_metrics(pl_module, trainer)

    def _process_validation(self, outputs, dataloader_idx):
        for key in outputs["metrics"]:
            n_value = outputs["metrics"][key]
            if isinstance(n_value, torch.Tensor):
                n_value = n_value.detach()

            if key not in self.val_metrics:
                if "histogram" in key:
                    self.val_metrics[dataloader_idx][key] = [n_value]
                elif "scatter" in key:
                    self.val_metrics[dataloader_idx][key] = [n_value[1]]
                    self.val_metrics_info[dataloader_idx][key] = n_value[0]
            else:
                if "histogram" in key:
                    self.val_metrics[dataloader_idx][key].append(n_value)
                elif "scatter" in key:
                    self.val_metrics[dataloader_idx][key].append(n_value[1])

    def _log_validation(self, outputs, pl_module, trainer, dataloader_idx):
        prefix = "" if self.val_names is None else f"{self.val_names[dataloader_idx]}_"
        for key in set(self.val_metrics[dataloader_idx]).union(outputs["metrics"]):
            logging_name = f"{prefix}val/{key}"
            if (
                "histogram" in key
                and len(self.val_metrics[dataloader_idx][key]) > 0
                and trainer.global_step + 1 % trainer.log_every_n_steps == 0
            ):
                hist = torch.cat(self.val_metrics[dataloader_idx][key])
                log_hist(trainer, logging_name, hist)
                self.val_metrics[dataloader_idx][key] = []
            elif (
                "scatter" in key
                and len(self.val_metrics[dataloader_idx][key]) > 0
                and trainer.global_step + 1 % trainer.log_every_n_steps == 0
            ):
                values = []
                for v in self.val_metrics[dataloader_idx][key]:
                    values += v.to("cpu").numpy().tolist()
                log_scatter(trainer, logging_name, values, self.val_metrics_info[dataloader_idx][key])
                self.val_metrics[dataloader_idx][key] = []
            elif not "histogram" in key and not "scatter" in key:
                pl_module.log(
                    logging_name, outputs["metrics"][key], on_epoch=True, add_dataloader_idx=False, rank_zero_only=True
                )

    @rank_zero_only
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Method called after each validation batch. This class is a Pytorch Lightning callback, therefore
        this method will by automatically called by Pytorch Lightning.

        Parameters
        ----------
        trainer: pl.Trainer
            Pytorch lightning trainer
        pl_module: pl.LightningModule
            Pytorch lightning module
        outputs:
            Training/Validation step outputs of the pl.LightningModule class.
            The `metrics` key is expected for this callback to work properly.
            m_outputs[`metrics`] must be a dict.
            For each key,
            - if the key contains the keyword 'histogram'
            The value of the tensor will be aggregate to compute an histogram all `trainer.log_every_n_steps`.
            - If the key contains the keyword 'scatter'
            The value of the key must of a tuple, (scatter_names: tuple, torch.tensor) with the first element
            being a list of len 2 with the name of the X axis and the Y axis. The second element will be
            a tensor of size (N, 2).
            - Othwerwise
            The tensor is expected to be a single scaler with the value to log
        batch: list
            Batch comming from the dataloader. Usually, a list of frame.
        batch_idx: int
            Id the batch
        dataloader_idx: int
            Dataloader id.
        """
        if trainer.logger is None:
            return
        if not isinstance(outputs, dict) or "metrics" not in outputs:
            raise Exception(
                "The lightning training_step must return a dict with the  `metrics` key to use this Callback"
            )

        self._process_validation(outputs, dataloader_idx)
        self._log_validation(outputs, pl_module, trainer, dataloader_idx)
