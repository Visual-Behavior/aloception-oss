from abc import ABC, abstractmethod
from typing import Union
import torch
import os
import warnings
import json
import yaml
from collections import OrderedDict
from typing import Tuple, Optional, Any

from alodataset.base_dataset import _user_prompt
from .helpers import (
    vb_folder,
    is_main_rank,
    is_dist_avail_and_initialized,
    get_expe_infos,
    get_latest_checkpoint_from_dir,
    get_best_checkpoint_from_dir,
    latest_cp_name,
    topk_cp_name,
)


class BaseTrainer(ABC):
    def __init__(
        self,
        project_name: str,
        experiment_name: str,
        val_interval: Union[int, float] = 1.0,
        log_interval: int = 50,
        save_best_k_cp: int = 3,
        no_suffix: bool = False,
    ):
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.val_interval = val_interval
        self.log_interval = log_interval
        self.no_suffix = no_suffix
        self.save_best_k_cp = save_best_k_cp
        self.current_step = 0
        self.checkpoint_infos = []

    def setup_workdir(self) -> None:
        """
        Setup the working directory
        """
        if (is_dist_avail_and_initialized() and is_main_rank()) or not is_dist_avail_and_initialized():
            vb_base_folder = vb_folder()
            os.makedirs(os.path.join(vb_base_folder, self.project_name), exist_ok=True)
            if os.path.exists(os.path.join(vb_base_folder, self.project_name, self.experiment_name)):
                warnings.warn(
                    f"Experiment {os.path.join(vb_base_folder, self.project_name, self.experiment_name)} already exists!."
                )
                warnings.warn(
                    "You may overwrite existing experiments. Ignore this message if you are resuming the run."
                )
                user_input = _user_prompt("Do you want to overwrite the existing experiment? (Y)es or (N)o: ")
                if user_input.lower() in ["y", "yes"]:
                    pass
                else:
                    raise ValueError("Experiment already exists. Please use a different name.")
            else:
                _, expe_dir, expe_name = get_expe_infos(self.project_name, self.experiment_name, self.no_suffix)
                self.experiment_name = expe_name
                os.makedirs(expe_dir)

    def get_latest_checkpoint_name(self, step: int = None) -> str:
        """
        Get the latest checkpoint name from the current step

        Args:
            step (int): step of the checkpoint

        Returns:
            str: latest checkpoint path
        """
        step = step if step is not None else self.current_step
        return latest_cp_name(step)

    def get_topk_checkpoint_name(self, metric_name: str, metric: float, step: int, epoch: int) -> str:
        """
        Get the topk checkpoint name from the current step

        Args:
            metric_name (str): name of the metric
            metric (float): value of the metric
            step (int): step of the checkpoint
            epoch (int): epoch of the checkpoint

        Returns:
            str: checkpoint path
        """
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

        expe_dir = os.path.join(vb_folder(), self.project_name, self.experiment_name)
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
        expe_dir = os.path.join(vb_folder(), self.project_name, self.experiment_name)
        if not os.path.exists(expe_dir):
            raise FileNotFoundError(f"Experiment {expe_dir} does not exist.")
        cp = get_latest_checkpoint_from_dir(expe_dir)
        if cp is None:
            raise FileNotFoundError(
                f"No latest checkpoint found. Latest checkpoint must have the format `latest_steps-<steps>` in {expe_dir}"
            )
        return cp

    def is_topk_checkpoint(
        self, epoch: int, metric_name: str, metric: float, condition: str = "min"
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Check if the checkpoint is in topk best checkpoints.
        If the checkpoint is in topk best checkpoints, return the new checkpoint path to save and the old checkpoint path to delete.

        Args:
            epoch (int): epoch of the checkpoint
            metric (str): name of the metric
            metric_value (float): value of the metric
            condition (str): condition to select the best checkpoint. `max` or `min`. Defaults to `min`
        Returns:
            Tuple[Optional[str], Optional[str]]: new checkpoint path to save and old checkpoint path to delete
        """
        assert condition in ["max", "min"], "condition must be either `max` or `min`"

        new_cp_dir = None  # Path to save the new checkpoint
        replaced_cp_dir = None  # This will be useful to remove the old checkpoint

        if self.save_best_k_cp == -1 or len(self.checkpoint_infos) < self.save_best_k_cp:
            self.checkpoint_infos.append(
                {"epoch": epoch, "step": self.current_step, "metric": metric, "metric_name": metric_name}
            )
            self.checkpoint_infos = sorted(self.checkpoint_infos, key=lambda x: x["metric"])
            new_cp_dir = self.get_topk_checkpoint_name(metric_name, metric, self.current_step, epoch)
        else:
            replaced_cp = None
            # Replace the worst checkpoint if the new checkpoint is better
            if condition == "min" and metric < self.checkpoint_infos[-1]["metric"]:
                replaced_cp = self.checkpoint_infos.pop(-1)
            elif condition == "max" and metric > self.checkpoint_infos[0]["metric"]:
                replaced_cp = self.checkpoint_infos.pop(0)
            if replaced_cp is not None:
                self.checkpoint_infos.append(
                    {
                        "epoch": epoch,
                        "step": self.current_step,
                        "metric": metric,
                        "metric_name": metric_name,
                    }
                )
                self.checkpoint_infos = sorted(self.checkpoint_infos, key=lambda x: x["metric"])
                new_cp_dir = self.get_topk_checkpoint_name(metric_name, metric, self.current_step, epoch)
                replaced_cp_dir = self.get_topk_checkpoint_name(
                    replaced_cp["metric_name"], replaced_cp["metric"], replaced_cp["epoch"], replaced_cp["step"]
                )

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
        self.checkpoint_infos = sorted(state["checkpoint_infos"], key=lambda x: x["metric"])
        self.current_step = state["current_step"]

    def save_checkpoint(
        self,
        cp_path: str,
        state_dicts: dict[str, OrderedDict],
    ) -> None:
        """
        Save the checkpoint and the checkpoint_info.json file in the checkpoint directory

        Args:
            cp_path (str): path to the checkpoint
            state_dicts (dict[str, OrderedDict]): state_dicts to save.
                The keys are the names of the state_dicts, and the values are the state_dicts.
                The state_dicts will be saved as `{key}.pth` in the checkpoint directory.
        """
        assert os.path.exists(cp_path), f"{cp_path} does not exist"

        for key, state_dict in state_dicts.items():
            torch.save(state_dict, os.path.join(cp_path, f"{key}.pth"))
        with open(os.path.join(cp_path, "state.json"), "w") as f:
            json.dump({"current_step": self.current_step, "checkpoint_infos": self.checkpoint_infos}, f)

    def resume_from_checkpoint(
        self,
        cp_path: str,
        state_dicts: dict[str, Any] = None,
        strict_loading: bool = True,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """
        Resume the trainer state and model/optimizer/lr_scheduler from the checkpoint

        Args:
            cp_path (str): path to the checkpoint
            state_dicts (dict[str, Any]): state_dicts to resume.
                The keys are the names of the state_dicts, and the values are the objects (model, optimizer,
                lr_scheduler, etc.) to resume.
                The state_dicts will be loaded from `{key}.pth` in the checkpoint directory.
            strict_loading (bool): if True, the state_dicts will be loaded with strict=True.
                If False, the state_dicts will be loaded with strict=False.
            device (torch.device): device to load the state_dicts.
        """
        assert os.path.exists(cp_path), f"{cp_path} does not exist"
        assert os.path.exists(os.path.join(cp_path, "state.json")), "state.json does not exist"

        self.load_trainer_state(cp_path)
        for key in state_dicts.keys():
            state_dicts[key].load_state_dict(
                torch.load(os.path.join(cp_path, f"{key}.pth"), map_location=device), strict=strict_loading
            )

    @abstractmethod
    def train(self):
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

    def test(self):
        """
        Test the model
        """
        warnings.warn("Test is not implemented for this trainer")
        pass
