import matplotlib.pyplot as plt
from pytorch_lightning.utilities import rank_zero_only
from alonet.common.logger import log_figure, log_scalar

from alonet.metrics import ApMetrics
from alonet.callbacks import BaseMetricsCallback


class ApMetricsCallback(BaseMetricsCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, base_metric=ApMetrics, **kwargs)

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        if trainer.logger is None:
            return

        for t, ap_metrics in enumerate(self.metrics):

            prefix = f"val/{t}/" if len(self.metrics) > 1 else "val/"
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

        self.metrics = []
