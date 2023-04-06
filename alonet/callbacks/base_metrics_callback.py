"""Class to implement a callback based for a specific metric

See Also
--------
    All the possible :doc:`alonet.metrics`
"""
import lightning as pl
import aloscene
from alonet import metrics
from pytorch_lightning.utilities import rank_zero_only

# import wandb


class InstancesBaseMetricsCallback(pl.Callback):
    """
    Parameters
    ----------
    base_metric : metrics
        A metric object of :doc:`alonet.metrics`
    """

    def __init__(self, base_metric: metrics, *args, **kwargs):
        self.metrics = []
        self.base_metric = base_metric
        super().__init__(*args, **kwargs)

    def inference(self, pl_module: pl.LightningModule, m_outputs: dict, **kwargs):
        """This method will call the :func:`inference` method of the module's model and will expect to receive the
        predicted boxes2D and/or Masks.

        Parameters
        ----------
        pl_module : pl.LightningModule
            Pytorch lighting module with inference function
        m_outputs : dict
            Forward outputs

        Returns
        -------
        :mod:`BoundingBoxes2D <aloscene.bounding_boxes_2d>`
            Boxes predicted
        :mod:`Mask <aloscene.mask>`
            Masks predicted

        Notes
        -----
            If :attr:`m_outputs` does not contain :attr:`pred_masks` attribute, a [None]*B list will be returned
            by default
        """
        b_pred_masks = None
        if "pred_masks" in m_outputs:
            b_pred_boxes, b_pred_masks = pl_module.inference(m_outputs, **kwargs)
        else:
            b_pred_boxes = pl_module.inference(m_outputs, **kwargs)
        if not isinstance(m_outputs, list):
            b_pred_boxes = [b_pred_boxes]
            b_pred_masks = [b_pred_masks]
        elif b_pred_masks is None:
            b_pred_masks = [None] * len(b_pred_boxes)
        return b_pred_boxes, b_pred_masks

    @rank_zero_only
    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: dict,
        batch: list,
        batch_idx: int,
        dataloader_idx: int,
    ):
        """Method call after each validation batch. This class is a pytorch lightning callback, therefore
        this method will by automaticly call by pl.

        This method will call the `infernece` method of the module's model and will expect to receive the
        predicted boxes2D and/or Masks. Theses elements will be aggregate to compute the different metrics in the
        `on_validation_end` method.
        The infernece method will be call using the `m_outputs` key from the outputs dict. If `m_outputs` is a list,
        then the list will be consider as an temporal list. Therefore, this callback will aggregate the prediction
        for each element of the sequence and will log the final results with the timestep prefix val/t/ instead of
        simply /val/

        Parameters
        ----------
        trainer: pl.Trainer
            Pytorch lightning trainer
        pl_module: pl.LightningModule
            Pytorch lightning module. The :attr:`m_outputs` key is expected for this this callback to work properly.
        outputs: dict
            Training/Validation step outputs of the pl.LightningModule class.
        batch: list
            Batch comming from the dataloader. Usually, a list of frame.
        batch_idx: int
            Id the batch
        dataloader_idx: int
            Dataloader batch ID.
        """
        if isinstance(batch, list):  # Resize frames for mask procedure
            batch = batch[0].batch_list(batch)

        b_pred_boxes, b_pred_masks = self.inference(pl_module, outputs["m_outputs"])
        is_temporal = isinstance(outputs["m_outputs"], list)
        for b, (t_pred_boxes, t_pred_masks) in enumerate(zip(b_pred_boxes, b_pred_masks)):

            # Retrieve the matching GT boxes at the same time step
            t_gt_boxes = batch[b].boxes2d
            t_gt_masks = batch[b].segmentation

            if not is_temporal:
                t_gt_boxes = [t_gt_boxes]
                t_gt_masks = [t_gt_masks]
            elif t_gt_masks is None:
                t_gt_masks = [None] * len(outputs["m_outputs"])

            if t_pred_masks is None:
                t_pred_masks = [None] if not is_temporal else [None] * len(outputs["m_outputs"])

            # Add the samples to metrics for each batch of the current sequence
            for t, (gt_boxes, pred_boxes, gt_masks, pred_masks) in enumerate(
                zip(t_gt_boxes, t_pred_boxes, t_gt_masks, t_pred_masks)
            ):
                if t + 1 > len(self.metrics):
                    self.metrics.append(self.base_metric())
                self.add_sample(self.metrics[t], pred_boxes, gt_boxes, pred_masks, gt_masks)

    @rank_zero_only
    def add_sample(
        self,
        base_metric: metrics,
        pred_boxes: aloscene.BoundingBoxes2D,
        gt_boxes: aloscene.BoundingBoxes2D,
        pred_masks: aloscene.Mask = None,
        gt_masks: aloscene.Mask = None,
    ):
        """Add a sample to some :doc:`alonet.metrics`. One might want to inhert this method
        to edit the :attr:`pred_boxes` and :attr:`gt_boxes` boxes before to add them.

        Parameters
        ----------
        base_metric : :doc:`alonet.metrics`
            Metric intance.
        pred_boxes : :mod:`BoundingBoxes2D <aloscene.bounding_boxes_2d>`
            Predicted boxes2D.
        gt_boxes : :mod:`BoundingBoxes2D <aloscene.bounding_boxes_2d>`
            GT boxes2d.
        pred_masks : :mod:`Mask <aloscene.mask>`
            Predicted Masks for segmentation task
        gt_masks : :mod:`Mask <aloscene.mask>`
            GT masks in segmentation task.
        """
        base_metric.add_sample(p_bbox=pred_boxes, t_bbox=gt_boxes, p_mask=pred_masks, t_mask=gt_masks)

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        """Method call at the end of each validation epoch. The method will use all the aggregate
        data over the epoch to log the final metrics on wandb.
        This class is a pytorch lightning callback, therefore this method will by automaticly call by pl.

        This method is currently a WIP since some metrics are not logged due to some wandb error when loading
        Table.

        Parameters
        ----------
        trainer: pl.Trainer
            Pytorch lightning trainer
        pl_module: pl.LightningModule
            Pytorch lightning module
        """
        if trainer.logger is None:
            return
        raise Exception("To inhert in a child class")
