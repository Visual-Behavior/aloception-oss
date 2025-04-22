import pytorch_lightning as pl
import numpy as np
import torch
import wandb

from typing import *
import aloscene


class FlowVideoCallback(pl.Callback):
    def __init__(self, datamodule, max_frames=30, fps=30):
        super().__init__()
        self.datamodule = datamodule
        self.names = datamodule.video_names
        self.loaders = datamodule.video_dataloader()
        self.max_frames = max_frames
        self.fps = fps
        if len(self.names) != len(self.loaders):
            raise Exception("Number of dataset names and images should not be different")

    def _init_val_frames(self):
        """
        Read the first image of each validation datasets
        """
        video_loaders = self.datamodule.video_dataloader()
        if not isinstance(video_loaders, list):
            video_loaders = [video_loaders]

        val_frames_list = []
        for loader in video_loaders:
            val_frames = next(iter(loader))
            if isinstance(val_frames, list):
                val_frames = aloscene.Frame.batch_list(val_frames)
            val_frames_list.append(val_frames)

        return val_frames_list

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.logger is None:
            return

        for name, loader in zip(self.names, self.loaders):
            video = self.create_video(loader, pl_module)

            trainer.logger.experiment.log({f"{name}/video/flow_pred": video}, step=trainer.global_step, commit=False)

    def create_video(self, loader, pl_module):
        iterator = iter(loader)
        snapshots = []
        for idx, frames in enumerate(iterator):
            if idx >= self.max_frames:
                break

            if isinstance(frames, list):
                frames = aloscene.Frame.batch_list(frames)

            if frames.device != pl_module.device:
                frames = frames.to(pl_module.device)

            flows = pl_module(frames, only_last=False)
            flows = pl_module.inference(flows, only_last=False)

            frames = frames.detach().cpu()
            flows = [flow.detach().cpu() for flow in flows]

            flow_pred = flows[-1]

            frame = frames[0][0]

            view = frame.get_view([frame]).add(flow_pred.get_view())
            snapshots.append(view.image.transpose((2, 0, 1)))

        snapshots = np.stack(snapshots, axis=0)
        snapshots = (snapshots * 255).astype(np.uint8)
        return wandb.Video(snapshots, fps=self.fps)
