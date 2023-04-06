import lightning as pl
import numpy as np
import torch
import wandb

from typing import *
import aloscene


def log_images_validation(frames: aloscene.Frame, trainer: pl.Trainer, name: str):

    assert tuple(frames.names) == ("B", "T", "C", "H", "W")
    assert frames.shape[0:2] == (1, 2)

    frames = frames[0]  # first batch
    for t in range(2):
        frame = frames[t]
        frame = frame.detach()
        frame = frame.norm255()
        frame = frame.cpu()
        frame = frame.type(torch.uint8)
        frame = frame.rename(None)
        frame = frame.permute([1, 2, 0])
        frame = frame.contiguous()
        frame = frame.numpy()

        wandb_img = wandb.Image(frame)

        wandb_name = f"{name}/images/t" if t == 0 else f"{name}/images/t+{t}"
        trainer.logger.experiment.log({wandb_name: wandb_img}, step=trainer.global_step, commit=False)


def log_flow_validation(
    frames: aloscene.Frame, flows_fw: List[aloscene.Flow], trainer: pl.Trainer, name: str
):
    """ """
    assert all(tuple(flow.names) == ("B", "C", "H", "W") for flow in flows_fw)
    assert all(flow.shape[0] == 1 for flow in flows_fw)
    assert tuple(frames.names) == ("B", "T", "C", "H", "W")
    assert frames.shape[0:2] == (1, 2)

    def get_flow_img(flow):
        flow_img = flow.__get_view__().image
        flow_img = (flow_img * 255).astype(np.uint8)
        flow_img = wandb.Image(flow_img)
        return flow_img

    # Log ground-truth
    flow_fw_gt = frames[0][0].flow["flow_forward"].detach().cpu()
    flow_fw_gt_img = get_flow_img(flow_fw_gt)
    trainer.logger.experiment.log(
        {f"{name}/flow/gt/flow_fw_gt": flow_fw_gt_img}, step=trainer.global_step, commit=False
    )

    # Log each iterations
    for it, flow_fw_pred in enumerate(flows_fw):
        flow_fw_pred = flow_fw_pred[0].detach().cpu()
        flow_fw_pred_img = get_flow_img(flow_fw_pred)
        trainer.logger.experiment.log(
            {f"{name}/flow/pred_iter/flow_fw_iter{it}": flow_fw_pred_img}, step=trainer.global_step, commit=False
        )

    # Log final flow
    flow_final = flows_fw[-1]
    flow_fw_pred = flow_final[0].detach().cpu()
    flow_fw_pred_img = get_flow_img(flow_fw_pred)
    trainer.logger.experiment.log(
        {f"{name}/flow/pred_final/flow_final": flow_fw_pred_img}, step=trainer.global_step, commit=False
    )


class RAFTFlowImagesCallback(pl.Callback):
    def __init__(self, datamodule):
        super().__init__()
        self.datamodule = datamodule
        self.names = datamodule.val_names
        self.val_frames_list = self._init_val_frames()
        if len(self.names) != len(self.val_frames_list):
            raise Exception("Number of dataset names and images should not be different")

    def _init_val_frames(self):
        """
        Read the first image of each validation datasets
        """
        val_loaders = self.datamodule.val_dataloader()
        if not isinstance(val_loaders, list):
            val_loaders = [val_loaders]

        val_frames_list = []
        for loader in val_loaders:
            val_frames = next(iter(loader))
            if isinstance(val_frames, list):
                val_frames = aloscene.Frame.batch_list(val_frames)
            val_frames_list.append(val_frames)

        return val_frames_list

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.logger is None:
            return

        for name, val_frames in zip(self.names, self.val_frames_list):

            if val_frames.device != pl_module.device:
                val_frames = val_frames.to(pl_module.device)

            flows = pl_module(val_frames, only_last=False)
            flows = pl_module.inference(flows, only_last=False)

            log_images_validation(val_frames, trainer, name)
            log_flow_validation(val_frames, flows, trainer, name)
