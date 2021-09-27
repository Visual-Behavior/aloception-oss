import aloscene
from typing import Union

from alonet.callbacks import ObjectDetectorCallback, ApMetricsCallback
from alonet import metrics
from pytorch_lightning.utilities import rank_zero_only
from alonet.detr_panoptic.utils import get_base_model_frame


class PanopticObjectDetectorCallback(ObjectDetectorCallback):
    """Panoptic Detr Callback for object detection training that use alonet.Frames as GT."""

    def __init__(self, val_frames: Union[list, aloscene.Frame]):
        # Batch list of frame if needed
        if isinstance(val_frames, list):
            val_frames = aloscene.Frame.batch_list(val_frames)
        super().__init__(val_frames=get_base_model_frame(val_frames))

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """ """
        if trainer.logger is None:
            return
        if trainer.fit_loop.should_accumulate() or (trainer.global_step + 1) % (trainer.log_every_n_steps * 10) != 0:
            # Draw boxes & masks for last batch
            if (trainer.fit_loop.total_batch_idx + 1) % trainer.num_training_batches != 0:
                return

        assert isinstance(outputs, dict)
        assert "m_outputs" in outputs

        pred_boxes, pred_masks = pl_module.inference(outputs["m_outputs"])
        frames = get_base_model_frame(batch)
        self.log_boxes_2d(frames=frames, preds_boxes=pred_boxes, trainer=trainer, name="train/frame_obj_detector")
        self.log_masks(frames=frames, pred_masks=pred_masks, trainer=trainer, name="train/frame_seg_detector")

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        """ """
        if trainer.logger is None:
            return

        # Send the validation frame on the same device than the Model
        if self.val_frames.device != pl_module.device:
            self.val_frames = self.val_frames.to(pl_module.device)

        pred_boxes, pred_masks = pl_module.inference(pl_module(self.val_frames))
        self.log_boxes_2d(
            frames=self.val_frames, preds_boxes=pred_boxes, trainer=trainer, name="val/frame_obj_detector"
        )
        self.log_masks(frames=self.val_frames, pred_masks=pred_masks, trainer=trainer, name="val/frame_seg_detector")


class PanopticApMetricsCallbacks(ApMetricsCallback):
    def add_sample(
        self,
        base_metric: metrics,
        pred_boxes: aloscene.BoundingBoxes2D,
        gt_boxes: aloscene.BoundingBoxes2D,
        pred_masks: aloscene.Mask = None,
        gt_masks: aloscene.Mask = None,
    ):
        if isinstance(gt_boxes.labels, dict):
            gt_boxes = gt_boxes.clone()
            gt_boxes.labels = gt_boxes.labels["category"]
        if isinstance(gt_masks.labels, dict):
            gt_masks = gt_masks.clone()
            gt_masks.labels = gt_masks.labels["category"]
        return super().add_sample(base_metric, pred_boxes, gt_boxes, pred_masks=pred_masks, gt_masks=gt_masks)
