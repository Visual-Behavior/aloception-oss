"""Callback that stores samples in each step to calculate the different Panoptic Quality metrics

See Also
--------
    :mod:`PQMetrics <alonet.metrics.compute_pq>`, the specific metric implement in this callback
"""
import matplotlib.pyplot as plt
from pytorch_lightning.utilities import rank_zero_only
from alonet.common.logger import log_figure, log_scalar

from alonet.metrics import PQMetrics
from alonet.callbacks import InstancesBaseMetricsCallback


class PQMetricsCallback(InstancesBaseMetricsCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, base_metric=PQMetrics, **kwargs)

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        if trainer.logger is None:
            return

        for t, pq_metrics in enumerate(self.metrics):

            prefix = f"val/{t}/" if len(self.metrics) > 1 else "val/"
            all_maps, all_maps_per_class = pq_metrics.calc_map(print_result=False)

            log_scalar(trainer, f"{prefix}PQ", all_maps["all"]["pq"])
            log_scalar(trainer, f"{prefix}SQ", all_maps["all"]["sq"])
            log_scalar(trainer, f"{prefix}RQ", all_maps["all"]["rq"])

            # Bar per each PQ class
            plt.style.use("ggplot")
            for cat in ["thing", "stuff"] if len(all_maps_per_class) > 1 else ["all"]:
                x_set, y_set = zip(*all_maps_per_class[cat].items())
                y_set = [y["pq"] for y in y_set]

                _, ax = plt.subplots()
                ax.barh(x_set, y_set)
                ax.set_xlabel("Panoptic Quality metric")
                ax.set_ylabel("Category")
                ax.set_title("PQ metric per category")
                log_figure(trainer, f"{prefix}pq_{cat}_per_class", plt.gcf())
                plt.clf()
                plt.cla()

        self.metrics = []
