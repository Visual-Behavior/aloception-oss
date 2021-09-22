import pytorch_lightning as pl
import aloscene
import alonet
import matplotlib.pyplot as plt
from pytorch_lightning.utilities import rank_zero_only
from alonet.common.logger import log_figure, log_scalar

# import wandb


class ApMetricsCallback(pl.Callback):
    def __init__(self, *args, **kwargs):
        self.ap_metrics = []
        super().__init__(*args, **kwargs)

    def inference(self, pl_module: pl.LightningModule, m_outputs: dict, **kwargs):
        b_pred_masks = None
        if "pred_masks" in m_outputs:
            b_pred_boxes, b_pred_masks = pl_module.inference(m_outputs, **kwargs)
        else:
            b_pred_boxes = pl_module.inference(m_outputs, **kwargs)
        if not isinstance(m_outputs, list):
            b_pred_boxes = [b_pred_boxes]
            b_pred_masks = [b_pred_masks]
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
        predicted boxes2D. Theses boxes will be aggregate to compute the AP metrics in the `on_validation_end` method.
        The infernece method will be call using the `m_outputs` key from the outputs dict. If `m_outputs` is a list,
        then the list will be consider as an temporal list. Therefore, this callback will aggregate the predicted boxes
        for each element of the sequence and will log the final results with the timestep prefix val/t/ instead of
        simply /val/

        Parameters
        ----------
        trainer: pl.Trainer
            Pytorch lightning trainer
        pl_module: pl.LightningModule
            Pytorch lightning module. The "m_outputs" key is expected for this this callback to work properly.
        outputs:
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

            if t_pred_masks is None:
                t_pred_masks = [None] * len(t_gt_masks)

            # Add the samples to to the AP metrics for each batch of the current sequence
            for t, (gt_boxes, pred_boxes, gt_masks, pred_masks) in enumerate(
                zip(t_gt_boxes, t_pred_boxes, t_gt_masks, t_pred_masks)
            ):
                if t + 1 > len(self.ap_metrics):
                    self.ap_metrics.append(alonet.metrics.ApMetrics())
                self.add_sample(self.ap_metrics[t], pred_boxes, gt_boxes, pred_masks, gt_masks)

    @rank_zero_only
    def add_sample(
        self,
        ap_metrics: alonet.metrics.ApMetrics,
        pred_boxes: aloscene.BoundingBoxes2D,
        gt_boxes: aloscene.BoundingBoxes2D,
        pred_masks: aloscene.Mask = None,
        gt_masks: aloscene.Mask = None,
    ):
        """Add a smple to some `alonet.metrics.ApMetrics()` class. One might want to inhert this method
        to edit the `pred_boxes` and `gt_boxes` boxes before to add them to the ApMetrics class.

        Parameters
        ----------
        ap_metrics: alonet.metrics.ApMetrics
            ApMetrics intance.
        pred_boxes: aloscene.BoundingBoxes2D
            Predicted boxes2D.
        gt_boxes: aloscene.BoundingBoxes2D
            GT boxes2d.
        pred_masks: aloscene.Mask
            Predicted Masks for segmentation task
        gt_masks: aloscene.Mask
            GT masks in segmentation task.
        """
        ap_metrics.add_sample(pred_boxes, gt_boxes, pred_masks, gt_masks)

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

        for t, ap_metrics in enumerate(self.ap_metrics):

            prefix = f"val/{t}/" if len(self.ap_metrics) > 1 else "val/"
            # step = trainer.global_step

            (
                all_maps,
                per_class_all_maps,
                all_maps_per_size,
                cross_clas_ap50_metrics,
                per_class_ap50_metrics,
            ) = ap_metrics.calc_map()

            # Precision/Confidence curve
            plt.plot(cross_clas_ap50_metrics["confidences"], cross_clas_ap50_metrics["precisions"], label="all")
            plt.xlabel("Confidence")
            plt.ylabel("Precision")
            plt.legend()
            plt.title("Precision/confidence curve")
            log_figure(trainer, f"{prefix}precision_confidence_curve_50", plt.gcf())
            plt.clf()
            plt.cla()

            # Recall/Confidence curve
            plt.plot(cross_clas_ap50_metrics["confidences"], cross_clas_ap50_metrics["recalls"], label="all")
            plt.xlabel("Confidence")
            plt.ylabel("Recall")
            plt.legend()
            plt.title("Recall/confidence curve")
            log_figure(trainer, f"{prefix}recall_confidence_curve_50", plt.gcf())
            plt.clf()
            plt.cla()

            # Roc curve: Recall vs Confidence curve
            plt.plot(cross_clas_ap50_metrics["recalls"], cross_clas_ap50_metrics["precisions"], label="all")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.legend()
            plt.title("ROC curve: Recall vs Precision")
            log_figure(trainer, f"{prefix}roc_curve_recall_vs_precision_50", plt.gcf())

            """
            cls_data = []
            cls_data.append(["box"] + [round(all_maps["box"][t], 2) for t in all_maps["box"]])
            cls_data.append(["mask"] + [round(all_maps["mask"][t], 2) for t in all_maps["mask"]])
            cls_data.append(["precision"] + [round(all_maps["precision"][t], 2) for t in all_maps["precision"]])
            cls_data.append(["recall"] + [round(all_maps["recall"][t], 2) for t in all_maps["recall"]])
            cls_table = wandb.Table(
                data=cls_data, columns=["Type"] + [str(threshold) for threshold in all_maps["box"]]
            )
            trainer.logger.experiment.log(
                {f"{prefix}table_map_scores": cls_table, "trainer/global_step": trainer.global_step}
            )

            cls_data = []
            cls_data.append(["box"] + [round(all_maps_per_size["box"][t], 2) for t in all_maps_per_size["box"]])
            cls_data.append(["mask"] + [round(all_maps_per_size["mask"][t], 2) for t in all_maps_per_size["mask"]])
            cls_data.append(
                ["precision"] + [round(all_maps_per_size["precision"][t], 2) for t in all_maps_per_size["precision"]]
            )
            cls_data.append(
                ["recall"] + [round(all_maps_per_size["recall"][t], 2) for t in all_maps_per_size["recall"]]
            )
            cls_data.append(
                ["box_ct"] + [round(all_maps_per_size["box_ct"][t], 2) for t in all_maps_per_size["box_ct"]]
            )
            cls_table = wandb.Table(
                data=cls_data, columns=["Type"] + [str(threshold) for threshold in all_maps_per_size["box"]]
            )
            trainer.logger.experiment.log(
                {f"{prefix}table_persize_scores": cls_table, "trainer/global_step": trainer.global_step}
            )

            # Per class bbox map50
            data = []
            for _cls in per_class_all_maps:
                data.append([_cls, round(per_class_all_maps[_cls]["box"][50], 2)])
            table = wandb.Table(data=data, columns=["cls_name", "bbox_map50"])
            trainer.logger.experiment.log(
                {f"{prefix}bar_perclass_bbox_map50": table, "trainer/global_step": trainer.global_step}
            )

            data = []
            for _cls in per_class_all_maps:
                data.append([_cls, round(per_class_all_maps[_cls]["precision"][50], 2)])
            table = wandb.Table(data=data, columns=["cls_name", "precision_map50"])
            trainer.logger.experiment.log(
                {f"{prefix}bar_perclass_precision_map50": table, "trainer/global_step": trainer.global_step}
            )

            data = []
            for _cls in per_class_all_maps:
                data.append([_cls, round(per_class_all_maps[_cls]["recall"][50], 2)])
            table = wandb.Table(data=data, columns=["cls_name", "recall_map50"])
            trainer.logger.experiment.log(
                {f"{prefix}bar_perclass_recall_map50": table, "trainer/global_step": trainer.global_step}
            )

            # Per class bbox map70
            data = []
            for _cls in per_class_all_maps:
                data.append([_cls, round(per_class_all_maps[_cls]["box"][70], 2)])
            table = wandb.Table(data=data, columns=["cls_name", "bbox_map70"])
            trainer.logger.experiment.log(
                {f"{prefix}bar_perclass_bbox_map70": table, "trainer/global_step": trainer.global_step}
            )

            data = []
            for _cls in per_class_all_maps:
                data.append([_cls, round(per_class_all_maps[_cls]["precision"][70], 2)])
            table = wandb.Table(data=data, columns=["cls_name", "precision_map70"])
            trainer.logger.experiment.log(
                {f"{prefix}bar_perclass_precision_map70": table, "trainer/global_step": trainer.global_step}
            )

            data = []
            for _cls in per_class_all_maps:
                data.append([_cls, round(per_class_all_maps[_cls]["recall"][70], 2)])
            table = wandb.Table(data=data, columns=["cls_name", "recall_map70"])
            trainer.logger.experiment.log(
                {f"{prefix}bar_perclass_recall_map70": table, "trainer/global_step": trainer.global_step}
            )
            """

            log_scalar(trainer, f"{prefix}map50_bbox", all_maps["box"][50])
            log_scalar(trainer, f"{prefix}map50_mask", all_maps["mask"][50])
            log_scalar(trainer, f"{prefix}map_bbox", all_maps["box"]["all"])
            log_scalar(trainer, f"{prefix}map_mask", all_maps["mask"]["all"])

        self.ap_metrics = []
