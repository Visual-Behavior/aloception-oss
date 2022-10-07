"""Detr Callback for object detection training that use :mod:`~aloscene.frame` as GT."""
import aloscene

from alonet.callbacks import ObjectDetectorCallback
from pytorch_lightning.utilities import rank_zero_only


class DetrObjectDetectorCallback(ObjectDetectorCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if trainer.logger is None:
            return
        if trainer.fit_loop._should_accumulate() or (trainer.global_step + 1) % (trainer.log_every_n_steps * 10) != 0:
            # Draw boxes for last batch
            if (trainer.fit_loop.total_batch_idx + 1) % trainer.num_training_batches != 0:
                return

        assert isinstance(outputs, dict)
        assert "m_outputs" in outputs

        if isinstance(batch, list):
            frames = aloscene.Frame.batch_list(batch)
        else:
            frames = batch

        pred_boxes = pl_module.inference(outputs["m_outputs"])
        self.log_boxes_2d(frames=frames, preds_boxes=pred_boxes, trainer=trainer, name="train/frame_obj_detector")

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.logger is None:
            return

        # Send the validation frame on the same device than the Model
        if self.val_frames.device != pl_module.device:
            self.val_frames = self.val_frames.to(pl_module.device)

        pred_boxes = pl_module.inference(pl_module(self.val_frames))
        self.log_boxes_2d(
            frames=self.val_frames, preds_boxes=pred_boxes, trainer=trainer, name="val/frame_obj_detector"
        )
