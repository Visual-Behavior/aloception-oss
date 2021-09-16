import torch
import pytorch_lightning as pl
import wandb
from typing import *
import aloscene
import alodataset
import alonet
from pytorch_lightning.utilities import rank_zero_only

import numpy as np
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from alonet.common.logger import log_image


class ObjectDetectorCallback(pl.Callback):
    """Callback for any object detection training that use
    alonet.Frames as GT.
    """

    def __init__(self, val_frames: Union[list, aloscene.Frame]):
        """The callback load frames every x training step as well as once
        every validation step on the given `val_frames`

        Parameters
        ----------
        val_frames: list | alonet.Frame
            List of sample from the validation set to use to load the validation progress
        """
        super().__init__()
        # Batch list of frame if needed
        if isinstance(val_frames, list):
            val_frames = aloscene.Frame.batch_list(val_frames)
        self.val_frames = val_frames

    def log_boxes_2d(self, frames: list, preds_boxes: list, trainer: pl.trainer.trainer.Trainer, name: str):
        """Given a frames and predicted boxes2d, this method will log the images into wandb

        Parameters
        ----------
        frames: list of aloscene.Frame
            Frame with GT boxes2d attached
        preds_boxes: list of aloscene.BoundingBoxes2D
            A set of predicted boxes2d
        trainer: pl.trainer.trainer.Trainer
            Lightning trainer
        """
        images = []
        for b, pred_boxes in enumerate(preds_boxes):

            # Retrive label names
            labels_names = frames.boxes2d[b].labels.labels_names
            if labels_names is not None:
                labels_names = {i: name for i, name in enumerate(labels_names)}

            if frames.boxes2d[b].padded_size is not None:
                pred_boxes = pred_boxes.as_boxes(frames.boxes2d[b])
                pred_boxes = pred_boxes.fit_to_padded_size()
                target_boxes = frames.boxes2d[b].fit_to_padded_size()
            else:
                target_boxes = frames.boxes2d[b]

            frame = frames[b]
            frame = frame.detach()
            frame = frame.norm255()
            frame = frame.cpu()
            frame = frame.type(torch.uint8)
            frame = frame.rename(None)
            frame = frame.permute([1, 2, 0])
            frame = frame.contiguous()
            frame = frame.numpy()

            # wandb_img = wandb.Image(frame, boxes=boxes)
            # images.append(wandb_img)
            images.append(
                {
                    "image": frame,
                    "boxes": [
                        {"name": "predictions", "class_labels": labels_names, "boxes": pred_boxes},
                        {"name": "ground_truth", "class_labels": labels_names, "boxes": target_boxes},
                    ],
                }
            )

        log_image(trainer, name, images)

        return

    def log_boxes_3d(
        self,
        frames: aloscene.Frame,
        preds_boxes: aloscene.BoundingBoxes3D,
        trainer: pl.trainer.trainer.Trainer,
        name: str,
    ):
        """Given a frames and predicted boxes3d, this method will log the images into wandb

        Parameter:
        ----------
        frames: aloscene.Frame
            Frame with GT boxes2d attached
        preds_boxes: aloscene.BoundingBoxes3D
            A set of predicted boxes3d
        trainer: pl.trainer.trainer.Trainer
            Lightning trainer
        """
        gt_images = []
        pred_images = []
        for b, boxes_3d in enumerate(preds_boxes):
            frame = frames[b]
            frame = frame.detach()
            pred_frame = frame.clone()
            target_frame = frame.clone()
            pred_frame.boxes3d = boxes_3d

            gt_view = target_frame.get_view([target_frame.boxes3d])
            pred_view = pred_frame.get_view([boxes_3d])
            gt_images.append({"image": wandb.Image((gt_view.image * 255).astype(np.uint8)), "boxes": None})
            pred_images.append({"image": wandb.Image((pred_view.image * 255).astype(np.uint8)), "boxes": None})

        log_image(trainer, name + "_gt", gt_images)
        log_image(trainer, name + "_pred", pred_images)
        return

    def log_masks(self, frames: list, pred_masks: list, trainer: pl.trainer.trainer.Trainer, name: str):
        """Given a frames and predicted masks in segmentation tasks, this method will log the images into wandb

        Parameters
        ----------
        frames: list of aloscene.Frame
            Frame with GT masks attached
        preds_masks: list of aloscene.Mask
            A set of predicted segmentation masks
        trainer: pl.trainer.trainer.Trainer
            Lightning trainer
        """
        images = []
        for b, p_mask in enumerate(pred_masks):

            # Retrive label names
            labels_names = frames.segmentation[b].labels.labels_names
            if labels_names is not None:
                labels_names = {i: name for i, name in enumerate(labels_names)}

            target_masks = frames.segmentation[b]

            frame = frames[b].detach().norm255().cpu().type(torch.uint8).rename(None)
            frame = frame.permute([1, 2, 0]).contiguous().numpy()

            # Get panoptic view
            target_masks = target_masks.masks2panoptic()
            p_mask = p_mask.masks2panoptic()
            target_masks[target_masks == -1] = 0  # N/A
            p_mask[p_mask == -1] = 0  # N/A
            target_masks = target_masks.astype(np.uint8)
            p_mask = p_mask.astype(np.uint8)

            # Add masks to frame
            masks = []
            if target_masks.size != 0:
                masks.append({"name": "ground_truth", "class_labels": labels_names, "masks": target_masks})
            if p_mask.size != 0:
                masks.append({"name": "predictions", "class_labels": labels_names, "masks": p_mask})
            images.append({"image": frame, "masks": masks if len(masks) > 0 else None})

        log_image(trainer, name, images)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """ """
        if trainer.logger is None:
            return
        raise Exception("To inhert in a child class")
        pass

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.logger is None:
            return
        raise Exception("To inhert in a child class")
