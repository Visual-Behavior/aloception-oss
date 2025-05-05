from abc import ABC, abstractmethod
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import os
import warnings
import json
from collections import OrderedDict
from typing import Tuple, Optional, Any, List, Dict
from safetensors.torch import save_file, load_file
from typing import Union

from alodataset.base_dataset import _user_prompt
from .helpers import (
    vb_folder,
    is_main_rank,
    get_expe_infos,
    get_latest_checkpoint_from_dir,
    get_expe_infos_from_checkpoint_path,
    get_best_checkpoint_from_dir,
    latest_cp_name,
    topk_cp_name,
    only_main_rank,
)
from .base_config import BaseConfig
from ..loggers import WandbLogger, TensorboardLogger, BaseLogger


class BaseTrainer(ABC):
    def __init__(
        self,
        project_name: str,
        experiment_name: str,
        accumulate_grad_batches: int = 1,
        num_epochs: Optional[int] = None,
        num_steps: Optional[int] = None,
        val_interval: Union[int, float] = 1.0,
        val_every_n_steps: int = None,
        log_interval: int = 50,
        save_best_k_cp: int = 3,
        logger: Optional[str] = "wandb",
        resume: bool = False,
        checkpoint: Optional[str] = None,
        no_suffix: bool = False,
        config: Optional[BaseConfig] = None,
    ):
        """
        BaseTrainer

        Parameters
        ----------
        project_name : str
            Name of the project
        experiment_name : str
            Name of the experiment
        accumulate_grad_batches : int
            Number of gradient accumulation steps
        num_epochs : int
            Number of epochs to train
        num_steps : int
            Number of steps to train
        val_interval : Union[int, float]
            Validation interval
        val_every_n_steps : int
            Run evaluation every n steps
        log_interval : int
            Log interval: number of steps between logging during training
        save_best_k_cp : int
            Number of best checkpoints to save
        logger : Optional[str]
            Logger to use
        resume : bool
            Resume training from the checkpoint
        checkpoint : Optional[str]
            Path to checkpoint file
        no_suffix : bool
            Whether to use no suffix for the experiment name
        config : Optional[BaseConfig]
            Configuration to log in experiment directory
        """
        assert logger is None or logger in ["wandb", "tensorboard"], "Only support `wandb` and `tensorboard`"
        assert num_epochs is not None or num_steps is not None, "Either `num_epochs` or `num_steps` must be set"

        self._accumulate_grad_batches = accumulate_grad_batches
        self._num_epochs = num_epochs
        self._num_steps = num_steps
        self._val_interval = val_interval
        self._val_every_n_steps = val_every_n_steps
        self._log_interval = log_interval
        self._no_suffix = no_suffix
        self._save_best_k_cp = save_best_k_cp
        self._resume = resume
        self._checkpoint = checkpoint
        self._current_step = 0
        self._current_epoch = 0
        self._checkpoint_infos = []
        self._config = config

        if is_main_rank():
            self._project_name, self._experiment_name = self.create_project_expe_name(
                project_name, experiment_name, no_suffix
            )
        else:
            self._project_name = None
            self._experiment_name = None

        # Setup the working directory
        self.setup_workdir()

        # Initialize logger
        self.logger = self.build_logger(logger)

    @property
    def project_name(self) -> str:
        """
        Project name

        Returns:
            str: Project name
        """
        return self._project_name

    @property
    def experiment_name(self) -> str:
        """
        Experience name

        Returns:
            str: Experience name

        """
        return self._experiment_name

    @property
    def accumulate_grad_batches(self) -> int:
        """
        Number of gradient accumulation step before back propagation

        Returns:
            int: Gradient accumulation step
        """
        return self._accumulate_grad_batches

    @property
    def val_interval(self) -> Union[int, float]:
        return self._val_interval

    @property
    def val_every_n_steps(self) -> int:
        """
        Run evaluation every n steps

        Returns:
            int: Number of steps to run evaluation
        """
        return self._val_every_n_steps

    @property
    def log_interval(self) -> int:
        """
        Log interval: number of steps between logging during training

        Returns:
            int: Log interval
        """
        return self._log_interval

    @property
    def save_best_k_cp(self) -> int:
        """
        Number of best checkpoints to save

        Returns:
            int: Number of best checkpoints to save
        """
        return self._save_best_k_cp

    @property
    def resume(self) -> bool:
        """
        Resume training from the checkpoint

        Returns:
            bool: Resume training from the checkpoint
        """
        return self._resume

    @property
    def no_suffix(self) -> bool:
        """
        Whether to use no suffix for the experiment name

        Returns:
            bool: Whether to use no suffix for the experiment name
        """
        return self._no_suffix

    @property
    def config(self) -> Optional[BaseConfig]:
        """
        Configuration to log in experiment directory

        Returns:
            Optional[BaseConfig]: Configuration
        """
        return self._config

    @property
    def checkpoint_infos(self) -> List[Dict[str, Any]]:
        """
        Checkpoint information. This is a list of dictionaries, each containing the information of a checkpoint.
        The information is saved in the `checkpoint_info.json` file in the checkpoint directory.

        Returns:
            List[Dict[str, Any]]: Checkpoint information
        """
        return self._checkpoint_infos

    @property
    def num_epochs(self) -> int:
        """
        Number of epochs to train

        Returns:
            int: Number of epochs to train
        """
        if self._num_epochs is None or self._num_epochs <= 0:
            Warning("num_epochs is not set for this training.")
        return self._num_epochs

    @num_epochs.setter
    def num_epochs(self, new_num_epochs: int) -> None:
        """
        Set the number of epochs to train

        Args:
            new_num_epochs (int): Number of epochs to train
        """
        self._num_epochs = new_num_epochs

    @property
    def num_steps(self) -> int:
        """
        Number of steps to train

        Returns:
            int: Number of steps to train
        """
        if self._num_steps is None or self._num_steps <= 0:
            Warning("num_steps is not set for this training.")
        return self._num_steps

    @property
    def current_epoch(self) -> int:
        """
        Current epoch

        Returns:
            int: Current epoch
        """
        return self._current_epoch

    @current_epoch.setter
    def current_epoch(self, new_current_epoch: int) -> None:
        """
        Set the current epoch

        Args:
            new_current_epoch (int): Current epoch
        """
        self._current_epoch = new_current_epoch

    @property
    def current_step(self) -> int:
        """
        Current step

        Returns:
            int: Current step
        """
        return self._current_step

    @current_step.setter
    def current_step(self, new_step: int) -> None:
        """
        Set the current step

        Args:
            new_step (int): Current step
        """
        self._current_step = new_step

    @property
    def checkpoint(self) -> Optional[str]:
        """
        Checkpoint path to resume training from

        Returns:
            Optional[str]: Checkpoint path
        """
        return self._checkpoint

    @checkpoint.setter
    def checkpoint(self, new_checkpoint: str) -> None:
        """
        Set the checkpoint path

        Args:
            new_checkpoint (str): Checkpoint path
        """
        self._checkpoint = new_checkpoint

    @property
    def is_training_done(self) -> bool:
        """
        Whether the training is done (either by reaching the number of steps or the number of epochs)

        Returns:
            bool: True if the training reaches the end, False otherwise
        """
        # If num_steps is set, check if the current step is greater than num_steps
        if self._num_steps is not None and self._num_steps > 0:
            return self._current_step >= self._num_steps

        # If num_steps is not set, check if the current epoch is greater than num_epochs
        return self._current_epoch >= self._num_epochs

    @only_main_rank
    def create_project_expe_name(self, project_name: str, experiment_name: str, no_suffix: bool) -> Tuple[str, str]:
        """
        Create the project name and experiment name

        Args:
            project_name (str): project name
            experiment_name (str): experiment name
            no_suffix (bool): if True, do not add a suffix to the experiment name

        Returns:
            Tuple[str, str]: project name and experiment name
        """
        if self._resume:
            assert self._checkpoint is not None, "Checkpoint path is not set"
            assert os.path.exists(self._checkpoint), f"{self._checkpoint} does not exist"
            project_dir, _, expe_name = get_expe_infos_from_checkpoint_path(self._checkpoint)
        else:
            project_dir, _, expe_name = get_expe_infos(project_name, experiment_name, no_suffix)
        return os.path.basename(project_dir), expe_name

    @only_main_rank
    def setup_workdir(self) -> None:
        """
        Setup the working directory
        """
        expe_dir = os.path.join(vb_folder(), self._project_name, self._experiment_name)
        if os.path.exists(expe_dir):
            warnings.warn(f"Experiment {expe_dir} exists!.")
            warnings.warn("You may overwrite existing experiments. Ignore this message if you are resuming the run.")
            user_input = _user_prompt("Do you want to overwrite the existing experiment? (Y)es or (N)o: ")
            if user_input.lower() in ["y", "yes"]:
                pass
            else:
                raise ValueError("Experiment already exists. Please use a different name.")
        else:
            print(f"Experiment {expe_dir} created.")
            os.makedirs(expe_dir)

        # Save config & command
        if self._config is not None:
            self._config.save(os.path.join(expe_dir, "config.yaml"))
        with open(os.path.join(expe_dir, "command.txt"), "w") as f:
            f.write(" ".join(sys.argv))

    @only_main_rank
    def build_logger(self, logger: Optional[str]) -> Optional[BaseLogger]:
        """
        Build the logger

        Args:
            logger (Optional[str]): Logger to use

        Returns:
            Optional[BaseLogger]: Logger
        """
        if logger is None:
            warnings.warn("No logger is chosen.")
            return None
        if logger == "wandb":
            return WandbLogger(
                name=self._experiment_name,
                project=self._project_name,
                save_dir=vb_folder(create_if_not_found=False),
                config=self._config.to_dict() if self._config is not None else None,
                resume=self._resume,
            )
        elif logger == "tensorboard":
            return TensorboardLogger(
                name=self._experiment_name,
                project=self._project_name,
                save_dir=vb_folder(create_if_not_found=False),
                resume=self._resume,
            )
        else:
            raise NotImplementedError(f"{logger} is not supported.")

    def get_latest_checkpoint_name(self, step: int = None) -> str:
        """
        Get the latest checkpoint name from the current step

        Args:
            step (int): step of the checkpoint

        Returns:
            str: latest checkpoint path
        """
        step = step if step is not None else self._current_step
        return latest_cp_name(step)

    def format_topk_checkpoint_name(
        self, metric_name: str, metric: float, step: int, epoch: Optional[int] = None
    ) -> str:
        """
        Get the topk checkpoint name from the current step

        Args:
            metric_name (str): name of the metric
            metric (float): value of the metric
            step (int): step of the checkpoint

        Returns:
            str: checkpoint path
        """
        epoch = epoch if epoch is not None else self._current_epoch
        return topk_cp_name(metric_name, metric, step, epoch)

    def get_best_checkpoint_path(self, condition: str = "max"):
        """
        Get the best checkpoint from the directory

        Args:
            dir_path (str): path to the directory
            condition (str): condition to select the best checkpoint. `max` or `min`. Defaults to `max`

        Returns:
            str: path to the best checkpoint
        """
        assert condition in ["max", "min"], "condition must be either `max` or `min`"

        expe_dir = os.path.join(vb_folder(), self._project_name, self._experiment_name)
        if not os.path.exists(expe_dir):
            raise FileNotFoundError(f"Experiment {expe_dir} does not exist.")

        cp = get_best_checkpoint_from_dir(expe_dir, condition)
        if cp is None:
            raise FileNotFoundError(
                f"No checkpoint of format `epoch=<epoch>_step=<step>_<metric>=<value>` found in {expe_dir}"
            )
        return cp

    def get_latest_checkpoint_path(self) -> str:
        """
        Get the latest checkpoint from the experiment directory
        """
        expe_dir = os.path.join(vb_folder(), self._project_name, self._experiment_name)
        if not os.path.exists(expe_dir):
            raise FileNotFoundError(f"Experiment {expe_dir} does not exist.")
        cp = get_latest_checkpoint_from_dir(expe_dir)
        if cp is None:
            raise FileNotFoundError(
                "No latest checkpoint found."
                f"Latest checkpoint must have the format `latest_steps-<steps>` in {expe_dir}"
            )
        return cp

    def is_topk_checkpoint(self, metric: float, condition: str = "min") -> bool:
        """
        Check if the checkpoint is in topk best checkpoints.

        Args:
            metric (float): value of the metric
            condition (str): condition to select the best checkpoitn. `max` or `min`. Defaults to `min`

        Returns:
            bool: True if the checkpoint is bestk
        """
        assert condition in ["min", "max"], "condition must be either `max` or `min"
        bestK = False
        if self._save_best_k_cp == -1 or len(self._checkpoint_infos) < self._save_best_k_cp:
            bestK = True
        else:
            if (condition == "min" and metric < self._checkpoint_infos[-1]["metric"]) or (
                condition == "max" and metric > self._checkpoint_infos[0]["metric"]
            ):
                bestK = True
        return bestK

    def get_checkpoint_dir(
        self, metric_name: str, metric: float, condition: str = "min"
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Update checkpoint information state and return the dir to save checkpoint and the dir of replaced checkpoint.

        Args:
            epoch (int): epoch of the checkpoint
            metric_name (str): name of the metric
            metric (float): value of the metric
            condition (str): condition to select the best checkpoint. `max` or `min`. Defaults to `min`
        Returns:
            Tuple[bool, Optional[str], Optional[str]]:
                - is current checkpoint in topK best checkpoints
                - new checkpoint path to save
                - old checkpoint path to replace
        """
        assert condition in ["max", "min"], "condition must be either `max` or `min`"

        new_cp_dir = None  # Path to save the new checkpoint
        replaced_cp_dir = None  # This will be useful to remove the old checkpoint

        if self._save_best_k_cp == -1 or len(self._checkpoint_infos) < self._save_best_k_cp:
            self._checkpoint_infos.append(
                {
                    "epoch": self._current_epoch,
                    "step": self._current_step,
                    "metric": metric,
                    "metric_name": metric_name,
                }
            )
            self._checkpoint_infos = sorted(self._checkpoint_infos, key=lambda x: x["metric"])
            new_cp_dir = self.format_topk_checkpoint_name(metric_name, metric, self._current_step)
            new_cp_dir = os.path.join(vb_folder(), self._project_name, self._experiment_name, new_cp_dir)
        else:
            replaced_cp = None
            # Replace the worst checkpoint if the new checkpoint is better
            if condition == "min" and metric < self._checkpoint_infos[-1]["metric"]:
                replaced_cp = self._checkpoint_infos.pop(-1)
            elif condition == "max" and metric > self._checkpoint_infos[0]["metric"]:
                replaced_cp = self._checkpoint_infos.pop(0)
            if replaced_cp is not None:
                self._checkpoint_infos.append(
                    {
                        "epoch": self._current_epoch,
                        "step": self._current_step,
                        "metric": metric,
                        "metric_name": metric_name,
                    }
                )
                self._checkpoint_infos = sorted(self._checkpoint_infos, key=lambda x: x["metric"])
                new_cp_dir = self.format_topk_checkpoint_name(metric_name, metric, self._current_step)
                new_cp_dir = os.path.join(vb_folder(), self._project_name, self._experiment_name, new_cp_dir)
                replaced_cp_dir = self.format_topk_checkpoint_name(
                    replaced_cp["metric_name"], replaced_cp["metric"], replaced_cp["step"], replaced_cp["epoch"]
                )
                replaced_cp_dir = os.path.join(vb_folder(), self._project_name, self._experiment_name, replaced_cp_dir)

        return new_cp_dir, replaced_cp_dir

    def load_trainer_state(self, cp_path: str) -> None:
        """
        Load state of Trainer

        Args:
            cp_path (str): path to the checkpoint
        """
        assert os.path.exists(cp_path), f"{cp_path} does not exist"
        with open(os.path.join(cp_path, "state.json"), "r") as f:
            state = json.load(f)
        self._checkpoint_infos = sorted(state["checkpoint_infos"], key=lambda x: x["metric"])
        self._current_step = state["current_step"]
        self._current_epoch = state["current_epoch"]

    @only_main_rank
    def save_checkpoint(
        self,
        cp_path: str,
        state_dicts: dict[str, OrderedDict],
        use_safetensors: dict[str, bool] = None,
    ) -> None:
        """
        Save the checkpoint and the checkpoint_info.json file in the checkpoint directory

        Args:
            cp_path (str): path to the checkpoint
            state_dicts (dict[str, OrderedDict]): state_dicts to save.
                The keys are the names of the modules, and the values are the state_dicts.
                The state_dicts will be saved as `{key}.pth` in the checkpoint directory.
            use_safetensors (dict[str, bool]): Use safetensors to save the state_dicts.
                The keys are the names of the modules, and the values are the boolean values.
                If True, the state_dicts will be saved as `{key}.safetensors` in the checkpoint directory.
                If False, use torch.save to save the state_dicts (pickle format).
                If `use_safetensors` is None, use torch.save to save the state_dicts (pickle format).
        """
        if not os.path.exists(cp_path):
            os.makedirs(cp_path)

        if use_safetensors is None:
            use_safetensors = {key: False for key in state_dicts.keys()}

        for key, state_dict in state_dicts.items():
            if use_safetensors[key]:
                save_file(state_dict, os.path.join(cp_path, f"{key}.safetensors"))
            else:
                torch.save(state_dict, os.path.join(cp_path, f"{key}.pth"))

        with open(os.path.join(cp_path, "state.json"), "w") as f:
            json.dump(
                {
                    "current_step": self._current_step,
                    "checkpoint_infos": self._checkpoint_infos,
                    "current_epoch": self._current_epoch,
                },
                f,
            )

    def resume_from_checkpoint(
        self,
        module: dict[str, Any],
        use_safetensors: dict[str, bool] = None,
        load_args: dict[str, dict[str, Any]] = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """
        Resume the trainer state and model/optimizer/lr_scheduler from the checkpoint

        Args:
            module (dict[str, Any]): module to resume.
                The keys are the names of the module, and the values are the objects (model, optimizer,
                lr_scheduler, etc.) to resume.
                The module will be loaded from `{key}.pth` in the checkpoint directory.
            strict_loading (bool): if True, the module will be loaded with strict=True.
                If False, the module will be loaded with strict=False.
            device (torch.device): device to load the module.
        """
        assert self._checkpoint is not None, "Checkpoint path is not set"
        assert self._resume, "Resume is not set"
        assert os.path.exists(self._checkpoint), f"{self._checkpoint} does not exist"
        assert os.path.exists(os.path.join(self._checkpoint, "state.json")), "state.json does not exist"

        self.load_trainer_state(self._checkpoint)

        # Set default load_args and use_safetensors
        if load_args is None:
            load_args = {key: {} for key in module.keys()}
        else:
            for key in module.keys():
                if key not in load_args.keys():
                    load_args[key] = {}

        if use_safetensors is None:
            use_safetensors = {key: False for key in module.keys()}
        else:
            for key in module.keys():
                if key not in use_safetensors.keys():
                    use_safetensors[key] = False

        # Load the module
        for key in module.keys():
            if use_safetensors[key]:
                if device == torch.device("cpu"):
                    device_str = "cpu"
                else:
                    device_str = "cuda"
                module[key].load_state_dict(
                    load_file(os.path.join(self._checkpoint, f"{key}.safetensors"), device=device_str),
                    **load_args[key],
                )
            else:
                module[key].load_state_dict(
                    torch.load(os.path.join(self._checkpoint, f"{key}.pth"), map_location=device),
                    **load_args[key],
                )
        print(f"Resumed from {self._checkpoint}")

    def reach_val_interval(self, dataloader_length: int) -> bool:
        """
        Check if the current step reaches validation step

        Args:
            dataloader_length (int): length of the dataloader
            accumulate_grad_batches (int): number of accumulate grad batches

        Returns:
            bool: True if the validation should be launched, False otherwise
        """
        # If val_every_n_steps is set, run evaluation every n steps
        if self._val_every_n_steps is not None:
            return (self._current_step + 1) % self._val_every_n_steps == 0

        # If val_every_n_steps is not set, run evaluation every val_interval
        eval_every_n_step = (
            self._val_interval
            if isinstance(self._val_interval, int)
            else dataloader_length // self._accumulate_grad_batches * self._val_interval
        )
        return (self._current_step + 1) % eval_every_n_step == 0

    def is_end_of_epoch(self, dataloader_length: int) -> bool:
        """
        Check if the current step reaches the end of the epoch

        Args:
            dataloader_length (int): length of the dataloader

        Returns:
            bool: True if the current step reaches the end of the epoch, False otherwise
        """
        return (self._current_step + 1) * self._accumulate_grad_batches % dataloader_length == 0

    @abstractmethod
    def build_criterion(self) -> nn.Module:
        """
        Build the criterion (Loss computation class)
        """
        pass

    @abstractmethod
    def build_optimizer(self) -> optim.Optimizer:
        """
        Build the optimizer
        """
        pass

    def build_lr_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        raise NotImplementedError

    @abstractmethod
    def training_step(self, *args, **kwargs):
        """
        One step of training
        """
        pass

    @abstractmethod
    def train(self, *args, **kwargs):
        """
        Train the model
        """
        pass

    @abstractmethod
    def validate(self):
        """
        Validate the model
        """
        pass

    @abstractmethod
    def validation_step(self):
        """
        One step of validation
        """
        pass
