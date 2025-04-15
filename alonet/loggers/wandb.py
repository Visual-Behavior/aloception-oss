import os
import wandb
from typing import Optional, Any, List
import json

from alonet.common.helpers import only_main_rank
from .base_logger import BaseLogger


class WandbLogger(BaseLogger):
    """Wandb logger."""

    def __init__(
        self,
        name: str,
        project: str,
        save_dir: str = ".",
        id: Optional[str] = None,
        group_name: Optional[str] = None,
        tags: Optional[list] = None,
        notes: Optional[str] = None,
        config: Optional[dict] = None,
        resume: bool = False,
    ):
        self.name = name
        self.project = project
        self.group_name = group_name
        self.tags = tags
        self.notes = notes
        self.config = config
        self.resume = resume
        self.id = id if id is not None else wandb.util.generate_id()
        self.save_dir = save_dir if save_dir is not None else os.path.join("wandb", self.project, self.name, "wandb")

        os.makedirs(self.save_dir, exist_ok=True)

        if self.resume:
            # get unique id of the run from wandb-id.json
            wandb_resume_path = os.path.join(self.save_dir, "wandb", "wandb-id.json")
            if os.path.exists(wandb_resume_path):
                with open(wandb_resume_path, "r") as f:
                    id = json.load(f)["run_id"]
            self.runner = wandb.init(project=self.project, name=self.name, resume=True, dir=self.save_dir, id=id)
        else:
            self.runner = wandb.init(
                project=self.project,
                name=self.name,
                group=self.group_name,
                tags=self.tags,
                notes=self.notes,
                config=self.config,
                resume="allow",
                id=self.id,
                dir=self.save_dir,
            )
            # get unique id of the run from wandb-id.json
            wandb_id_path = os.path.join(self.save_dir, "wandb", "wandb-id.json")
            with open(wandb_id_path, "w") as f:
                json.dump({"run_id": self.id}, f)

        # Set step
        wandb.define_metric("train/global_step")
        wandb.define_metric("train/*", step_metric="train/global_step")
        wandb.define_metric("val/*", step_metric="train/global_step")

    @only_main_rank
    def log_scalar(self, step: int, key: str, obj: Any):
        """Log a scalar to the current logger

        Parameters
        ----------
        step: int
            The current step
        key: str
            Tag name of the log
        obj: Any
            The scalar to log
        """
        self.runner.log({key: obj, "train/global_step": step})

    @only_main_rank
    def log_image(self, step: int, key: str, image: Any):
        """Log an image to the current logger

        Parameters
        ----------
        step: int
            The current step
        key: str
            Tag name of the log
        image: Any
            The image to log
        """
        self.runner.log({key: wandb.Image(image), "train/global_step": step})

    @only_main_rank
    def log_scatter(self, step: int, key: str, values: Any, column_names: List[str]):
        """Log a scatter graph to the current logger

        Parameters
        ----------
        step: int
            The current step
        key: str
            Tag name of the log
        values: torch.Tensor
            values of the graphe tensor (N, 2)
        column_names:List[str]
            names of axes (x axis, y axis)
        """
        self.runner.log(
            {
                key: wandb.plot.scatter(
                    wandb.Table(data=values, columns=column_names), x=column_names[0], y=column_names[1], title=key
                ),
                "train/global_step": step,
            }
        )

    @only_main_rank
    def log_hist(self, step: int, key: str, hist: Any):
        """Log a histogram graph to the current logger

        Parameters
        ----------
        step: int
            The current step
        key: str
            Tag name of the log
        hist: torch.tensor
            names of axes (x axis, y axis)
        """
        self.runner.log({key: wandb.Histogram(hist), "train/global_step": step})
