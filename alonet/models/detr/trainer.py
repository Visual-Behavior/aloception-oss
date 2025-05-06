import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.distributed as dist
import shutil
from typing import Union, Tuple, List
from tqdm import tqdm
import cv2
import numpy as np

import aloscene
from aloscene import Frame
from alonet.common import (
    BaseTrainer,
    is_main_rank,
    only_main_rank,
    is_dist_avail_and_initialized,
    setup_data_fetcher,
    get_model_state_dict,
    get_rank,
)

from .models import DetrR50, DetrHungarianMatcher
from .criterion import DetrCriterion


class DetrTrainer(BaseTrainer):
    def __init__(self, model_name: str, weights: str, viz_interval: int = 500, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.weights = weights
        self.viz_interval = viz_interval
        self.model: nn.Module = self.build_model()
        self.criterion = self.build_criterion()
        self.optimizer = self.build_optimizer()
        self.device = torch.device(f"cuda:{get_rank()}" if torch.cuda.is_available() else "cpu")

    def build_model(self, num_classes: int = 91, aux_loss: bool = True, weights: str = None) -> nn.Module:
        """Build the default model

        Parameters
        ----------
        num_classes : int, optional
            Number of classes in embed layer, by default 91
        aux_loss : bool, optional
            Return auxiliar outputs in forward output, by default True
        weights : str, optional
            Path or id to load weights, by default None

        Returns
        -------
        :mod:`~alonet.models.detr.models.detr`
            Pytorch model

        Raises
        ------
        Exception
            Only :attr:`detr-r50` models are supported yet.
        """
        if self.model_name == "detr-r50":
            return DetrR50(num_classes=num_classes, aux_loss=aux_loss, weights=self.weights)
        else:
            raise Exception(f"Unsupported base model {self.model_name}")

    def build_matcher(
        self, cost_class: float = 1, cost_boxes: float = 5, cost_giou: float = 2
    ) -> DetrHungarianMatcher:
        """Build the default matcher

        Parameters
        ----------
        cost_class : float, optional
            Weight of class cost in Hungarian Matcher, by default 1
        cost_boxes : float, optional
            Weight of boxes cost in Hungarian Matcher, by default 5
        cost_giou : float, optional
            Weight of GIoU cost in Hungarian Matcher, by default 2

        Returns
        -------
        :mod:`DetrHungarianMatcher <alonet.models.detr.matcher>`
            Hungarian Matcher, as a Pytorch model
        """
        return DetrHungarianMatcher(cost_class=cost_class, cost_boxes=cost_boxes, cost_giou=cost_giou)

    def build_criterion(
        self,
        loss_ce_weight=1,
        loss_boxes_weight=5,
        loss_giou_weight=2,
        eos_coef=0.1,
        losses=["labels", "boxes"],
        aux_loss_stage=6,
    ) -> DetrCriterion:
        """Build the default criterion

        Parameters
        ----------
        matcher : torch.nn, optional
            One specfic matcher to use in criterion process, by default the output of :func:`build_matcher`
        loss_ce_weight : float, optional
            Weight of cross entropy loss in total loss, by default 1
        loss_boxes_weight : float, optional
            Weight of boxes loss in total loss, by default 5
        loss_giou_weight : float, optional
            Weight of GIoU loss in total loss, by default 2
        eos_coef : float, optional
            Background/End of the Sequence (EOS) coefficient, by default 0.1
        losses : list, optional
            List of losses to take into account in total loss, by default ["labels", "boxes"].
            Possible values: ["labels", "boxes", "masks"] (use the latest in segmentation tasks)
        aux_loss_stage : int, optional
            Size of stages from :attr:`aux_outputs` key in forward ouputs, by default 6

        Returns
        -------
        :mod:`DetrCriterion <alonet.detr.criterion>`
            Criterion use to train the model
        """
        matcher = self.build_matcher()
        return DetrCriterion(
            matcher=matcher,
            loss_ce_weight=loss_ce_weight,
            loss_boxes_weight=loss_boxes_weight,
            loss_giou_weight=loss_giou_weight,
            eos_coef=eos_coef,
            losses=losses,
            aux_loss_stage=aux_loss_stage,
        )

    def build_optimizer(self) -> torch.optim.Optimizer:
        """AdamW optimizer configuration, using different learning rates for backbone and others parameters

        Returns
        -------
        torch.optim
            `AdamW <https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html>`_ optimizer to update weights
        """
        param_dicts = [
            {"params": [p for n, p in self.model.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.model.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": 1e-5,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=1e-4, weight_decay=1e-4)
        return optimizer

    def assert_input(self, frames: Frame, inference=False):
        """Check if input-frames have the correct format

        Parameters
        ----------
        frames : :mod:`Frames <aloscene.frame>`
            Input frames
        inference : bool, optional
            Check input from inference procedure, by default False
        """
        assert isinstance(frames, aloscene.Frame)
        assert frames.normalization == "resnet", f"{frames.normalization}"
        assert frames.mean_std[0] == self.model.INPUT_MEAN_STD[0]
        assert frames.mean_std[1] == self.model.INPUT_MEAN_STD[1]
        assert frames.names == ("B", "C", "H", "W"), f"{frames.names}"
        if not inference:
            assert frames.boxes2d is not None
            assert frames.mask is not None
            assert frames.mask.names == ("B", "C", "H", "W")

    def forward(self, frames: Union[list, Frame], **kwargs) -> dict:
        """Run a forward pass through the model.

        Parameters
        ----------
        frames : Union[list, :mod:`Frames <aloscene.frame>`]
            List of :mod:`~aloscene.frame` without batch dimension or a Frame with the batch dimension

        Returns
        -------
        m_outputs: dict
            A dict with the forward detr outputs.
        """
        # Batch list of frame if needed
        if isinstance(frames, list):
            frames = aloscene.Frame.batch_list(frames)
        # Assert inputs content
        self.assert_input(frames, inference=True)
        # Run forward pass
        m_outputs = self.model(frames, **kwargs)
        return m_outputs

    def inference(self, m_outputs: dict, **kwargs):
        """Given the model forward outputs, this method will return an
        :mod:`BoundingBoxes2D <aloscene.bounding_boxes_2d>` tensor.

        Parameters
        ----------
        m_outputs: dict
            Dict with the model forward outptus

        Returns
        -------
        List[:mod:`BoundingBoxes2D <aloscene.bounding_boxes_2d>`]
            Set of boxes for each batch
        """
        return self.model.inference(m_outputs, **kwargs)

    def common_step(self, frames: Union[list, Frame], compute_statistical_metrics: bool) -> Tuple[torch.Tensor, dict]:
        # Batch list of frame if needed
        if isinstance(frames, list):
            frames = aloscene.Frame.batch_list(frames)

        frames = frames.to(self.device)

        # Assert inputs content
        self.assert_input(frames)
        m_outputs = self.model(frames)

        total_loss, losses = self.criterion(m_outputs, frames, compute_statistical_metrics=compute_statistical_metrics)

        def detach_all(var):
            if isinstance(var, torch.Tensor):
                return var.detach()
            elif isinstance(var, dict):
                for k, v in var.items():
                    var[k] = detach_all(v)
            elif isinstance(var, list):
                for i, v in enumerate(var):
                    var[i] = detach_all(v)
            return var

        outputs = {}
        outputs.update({"metrics": detach_all(losses)})
        outputs.update({"m_outputs": detach_all(m_outputs)})
        return total_loss, outputs

    @only_main_rank
    def log_metrics(self, metrics: dict) -> None:
        if self.logger is not None:
            for k, v in metrics.items():
                self.logger.log_scalar(self.current_step, k, v)

    @only_main_rank
    def log_visualization(self, frames: List[Frame], outputs: dict, stage: str) -> None:
        if self.logger is not None:
            frame = frames[0]
            boxes = self.inference(outputs["m_outputs"])

            gt_image = frame.get_view().image
            gt_image = (gt_image * 255).astype(np.uint8)
            det_frame = frame.clone()
            det_frame.boxes2d = boxes[0]
            det_image = det_frame.get_view().image
            det_image = (det_image * 255).astype(np.uint8)
            self.logger.log_image(self.current_step, f"images_{stage}/gt", gt_image)
            self.logger.log_image(self.current_step, f"images_{stage}/detection", det_image)

    def save_checkpoint_if_topk(self, loss: float) -> None:
        if self.is_topk_checkpoint(loss, "min"):
            new_cp_path, old_cp_path = self.get_checkpoint_dir("loss", loss, "min")
            if new_cp_path is not None:
                print(f"Saving checkpoint at {new_cp_path}")
                model_state_dict = get_model_state_dict(self.model)
                self.save_checkpoint(
                    new_cp_path,
                    state_dicts={"model": model_state_dict, "optimizer": self.optimizer.state_dict()},
                    use_safetensors={"model": True, "optimizer": False},
                )
                if old_cp_path is not None:
                    print(f"Removing checkpoint at {old_cp_path}")
                    shutil.rmtree(old_cp_path)

    @torch.no_grad()
    def validation_step(
        self, frames: Union[list, Frame], compute_statistical_metrics: bool
    ) -> Tuple[torch.Tensor, dict]:
        """Run one step of validation

        Parameters
        ----------
        frames : Union[list, :mod:`Frames <aloscene.frame>`]
            List of :mod:`~aloscene.frame` without batch dimension or a Frame with the batch dimension
        batch_idx : int
            Batch id given by Lightning

        Returns
        -------
        dict
            Dictionary with the :attr:`loss` to optimize, :attr:`m_outputs` forward outputs and :attr:`metrics` to log.
        """
        return self.common_step(frames, compute_statistical_metrics)

    def training_step(
        self, frames: Union[list, Frame], compute_statistical_metrics: bool, no_sync: bool = True
    ) -> Tuple[torch.Tensor, dict]:
        """Train the model for one step

        Parameters
        ----------
        frames : Union[list, :mod:`Frames <aloscene.frame>`]
            List of :mod:`~aloscene.frame` without batch dimension or a Frame with the batch dimension
        batch_idx : int
            Batch id given by Lightning

        Returns
        -------
        dict
            Dictionary with the :attr:`loss` to optimize, :attr:`m_outputs` forward outputs and :attr:`metrics` to log.
        """
        loss, outputs = self.common_step(frames, compute_statistical_metrics)
        loss /= self.accumulate_grad_batches
        loss.backward()
        return loss, outputs

    @torch.no_grad()
    def validate(self, val_dataloader: DataLoader) -> dict:
        progress_bar = tqdm(total=len(val_dataloader))
        desc = "Step {current_step}/{total_steps}"

        metrics = {"val/total_loss": []}
        for i, frames in enumerate(val_dataloader):
            loss, outputs = self.common_step(frames, compute_statistical_metrics=False)
            metrics["val/total_loss"].append(loss.item())
            for k, v in outputs["metrics"].items():
                if k not in metrics:
                    metrics[f"val/{k}"] = []
                metrics[f"val/{k}"].append(v)
            progress_bar.update(1)
            progress_bar.set_description(desc.format(current_step=i, total_steps=len(val_dataloader)))

        for k, v in metrics.items():
            metrics[k] = torch.mean(torch.tensor(v)).item()
        return metrics

    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader) -> None:
        main_rank = is_main_rank()

        # Init progress bar
        if main_rank:
            progress_bar = tqdm(total=len(train_dataloader))
            desc = "Epoch {current_epoch}/{total_epochs} - Step {current_step}/{total_steps} - Loss {loss:.4f}"

        if self.num_epochs is None:
            self.num_epochs = self.num_steps * self.accumulate_grad_batches // len(train_dataloader)

        train_metrics = {"train/total_loss": []}
        data_fetcher = setup_data_fetcher(train_dataloader)

        while not self.is_training_done:
            if self.is_end_of_epoch(len(train_dataloader)):
                self.current_epoch += 1
                if main_rank:
                    progress_bar.reset()

            for i in range(self.accumulate_grad_batches):
                frames = next(data_fetcher)
                loss, outputs = self.training_step(frames, False)
                if main_rank:
                    train_metrics["train/total_loss"].append(loss.detach().item())
                    for k, v in outputs["metrics"].items():
                        if k not in train_metrics:
                            train_metrics[f"train/{k}"] = []
                        train_metrics[f"train/{k}"].append(v)
                    progress_bar.update(1)
                    progress_bar.set_description(
                        desc.format(
                            current_epoch=self.current_epoch,
                            total_epochs=self.num_epochs,
                            current_step=(self.current_step * self.accumulate_grad_batches + i)
                            % len(train_dataloader),
                            total_steps=len(train_dataloader),
                            loss=loss.detach(),
                        )
                    )

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.current_step % self.log_interval:
                for k, v in train_metrics.items():
                    train_metrics[k] = torch.mean(torch.tensor(v)).item()
                self.log_metrics(train_metrics)
                train_metrics = {"train/total_loss": []}

            if self.current_step % self.viz_interval == 0:
                self.log_visualization(frames, outputs, "train")

            if self.reach_val_interval(len(train_dataloader)):
                # Sync all processes before validation in case of multi-GPU training
                if is_dist_avail_and_initialized():
                    dist.barrier()

                if main_rank:
                    metrics = self.validate(val_dataloader)
                    self.save_checkpoint_if_topk(metrics["val/total_loss"])
                    self.log_metrics(metrics)
                    self.model.train()

                if is_dist_avail_and_initialized():
                    dist.barrier()

            self.current_step += 1

        # Save last checkpoint
        print("Training is done ! Saving last checkpoint")
        self.save_checkpoint(
            self.get_latest_checkpoint_path(),
            {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict()},
            use_safetensors=True,
        )
