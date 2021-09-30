"""Compute Panoptic, Segmentation and Recognition Qualities Metrics.

Examples
--------
.. code-block:: python

    import torch

    from alodataset import CocoPanopticDataset, Split
    from alonet.detr_panoptic import LitPanopticDetr
    from alonet.metrics import PQMetrics

    from aloscene import Frame

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # Model/Dataset definition to evaluate
    dataset = CocoPanopticDataset(split=Split.VAL)
    model = LitPanopticDetr(weights = "detr-r50-panoptic").model
    model.eval().to(device)

    # Metric to develop
    metric = PQMetrics()

    for frame in dataset.stream_loader():
        # Frame to batch
        frame = Frame.batch_list([frame]).to(device)

        # Masks inference and get GT from frame
        _, pred_masks = model.inference(model(frame))
        pred_masks = pred_masks[0] # Predictions of the first batch
        gt_masks = frame[0].segmentation  # GT of first batch

        # Add samples to evaluate metrics
        metric.add_sample(p_mask=pred_masks, t_mask=gt_masks)

    # Print results
    metric.calc_map(print_result=True)

.. list-table:: Summary of expected result
    :widths: 15 15 10 10 10
    :header-rows: 1
    :align: center

    * - Category
      - Total cat.
      - PQ
      - SQ
      - RQ
    * - Things
      - 80
      - 0.557
      - 0.803
      - 0.685
    * - Stuff
      - 53
      - 0.370
      - 0.784
      - 0.463
    * - All
      - 133
      - 0.483
      - 0.795
      - 0.596
"""

# Inspired by the official panopticapi and adapted for aloception
# https://github.com/cocodataset/panopticapi/blob/master/panopticapi/evaluation.py

import numpy as np
import torch
from collections import defaultdict
from typing import Dict

import aloscene

VOID = -1
OFFSET = 256 * 256 * 256


class PQStatCat(object):
    """Keep TP, FP, FN and IoU metrics per class"""

    def __init__(self):
        self.iou = 0.0
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def __iadd__(self, pq_stat_cat):
        self.iou += pq_stat_cat.iou
        self.tp += pq_stat_cat.tp
        self.fp += pq_stat_cat.fp
        self.fn += pq_stat_cat.fn
        return self


class PQMetrics(object):
    def __init__(self):
        self.pq_per_cat = defaultdict(PQStatCat)
        self.class_names = None
        self.categories = dict()

    def __getitem__(self, label_id: int):
        return self.pq_per_cat[label_id]

    def __iadd__(self, pq_stat):
        for label, pq_stat_cat in pq_stat.pq_per_cat.items():
            self.pq_per_cat[label] += pq_stat_cat
        return self

    def update_data_objects(self, cat_labels: aloscene.Labels, isthing_labels: aloscene.Labels):
        """Update data objects categories, appending new categories from each sample

        Parameters
        ----------
        cat_labels : :mod:`Labels <aloscene.labels>`
            Categories labels to append
        isthing_labels : :mod:`Labels <aloscene.labels>`
            Description of thing/stuff labels to append
        """
        self.class_names = cat_labels.labels_names
        self.categories.update(
            {
                id: {"category": self.class_names[id], "isthing": it == 1}
                for id, it in zip(list(cat_labels.numpy().astype(int)), list(isthing_labels.numpy().astype(int)))
            }
        )

    def pq_average(self, isthing: bool = None, print_result: bool = False):
        """Calculate SQ, RQ and PQ metrics from the categories, and thing/stuff/all if desired

        Parameters
        ----------
        isthing : bool
            Calculate metrics for the 'thing' category (if it True) or 'stuff' category (if it False).
            By default the metric is executed over both
        print_result : bool
            Print result in console, by default False

        Returns
        -------
        Tuple[Dict, Dict]
            Dictionaries with general PQ average metrics and for each class, respectively
        """
        pq, sq, rq, n = 0, 0, 0, 0
        per_class_results = {}
        for label, label_info in self.categories.items():
            if isthing is not None:
                cat_isthing = label_info["isthing"]
                if isthing != cat_isthing:
                    continue
            iou = self.pq_per_cat[label].iou
            tp = self.pq_per_cat[label].tp
            fp = self.pq_per_cat[label].fp
            fn = self.pq_per_cat[label].fn
            if tp + fp + fn == 0:
                per_class_results[label_info["category"]] = {"pq": 0.0, "sq": 0.0, "rq": 0.0}
                continue
            n += 1
            pq_class = iou / (tp + 0.5 * fp + 0.5 * fn)
            sq_class = iou / tp if tp != 0 else 0
            rq_class = tp / (tp + 0.5 * fp + 0.5 * fn)
            per_class_results[label_info["category"]] = {"pq": pq_class, "sq": sq_class, "rq": rq_class}
            pq += pq_class
            sq += sq_class
            rq += rq_class

        result = {"pq": pq / n, "sq": sq / n, "rq": rq / n, "n": n}
        if print_result:
            suffix = ""
            if isthing is not None and isthing:
                suffix = "th"
            elif isthing is not None and not isthing:
                suffix = "st"
            _print_map(result, per_class_results, suffix=suffix)
        return result, per_class_results

    def add_sample(
        self, p_mask: aloscene.Mask, t_mask: aloscene.Mask, **kwargs,
    ):
        """Add a new prediction and target masks to PQ metrics estimation process

        Parameters
        ----------
        p_mask : :mod:`Mask <aloscene.mask>`
            Predicted masks by network inference
        t_mask : :mod:`Mask <aloscene.mask>`
            Target masks with labels and labels_names properties

        Raises
        ------
        Exception
            :attr:`p_mask` and :attr:`t_mask` must be an :mod:`Mask <aloscene.mask>` object and have the same shapes,
            as well as must have labels attribute. Finally, :attr:`t_mask` must have two minimal labels:
            :attr:`category` and :attr:`isthing`
        """
        assert isinstance(p_mask, aloscene.Mask) and isinstance(t_mask, aloscene.Mask)
        assert isinstance(p_mask.labels, aloscene.Labels) and isinstance(t_mask.labels, dict)
        assert "category" in t_mask.labels and "isthing" in t_mask.labels
        assert hasattr(t_mask.labels["category"], "labels_names") and hasattr(t_mask.labels["isthing"], "labels_names")
        assert len(t_mask.labels["category"]) == len(t_mask.labels["isthing"])

        p_mask = p_mask.to(torch.device("cpu"))
        t_mask = t_mask.to(torch.device("cpu"))

        self.update_data_objects(t_mask.labels["category"], t_mask.labels["isthing"])

        pan_pred = p_mask.mask2id()
        pan_gt = t_mask.mask2id(labels_set="category")

        # ground truth segments area calculation
        gt_segms = {}
        labels, labels_cnt = np.unique(pan_gt, return_counts=True)
        for label, label_cnt in zip(labels, labels_cnt):
            if label == VOID:  # Ignore pixels without category
                continue
            assert label < len(self.class_names)
            gt_segms[label] = label_cnt  # Get area for each class

        # predicted segments area calculation
        pred_segms = {}
        labels, labels_cnt = np.unique(pan_pred, return_counts=True)
        for label, label_cnt in zip(labels, labels_cnt):
            if label == VOID:  # Ignore pixels without category
                continue
            assert label < len(self.class_names)
            pred_segms[label] = label_cnt  # Get area for each class

        # confusion matrix calculation if not empty views
        gt_pred_map = {}
        if len(gt_segms) > 0 and len(pred_segms) > 0:
            aux_off = VOID if VOID < 0 else 0
            pan_gt = pan_gt - aux_off
            pan_pred = pan_pred - aux_off

            pan_gt_pred = pan_gt.astype(np.uint64) * OFFSET + pan_pred.astype(np.uint64)
            labels, labels_cnt = np.unique(pan_gt_pred, return_counts=True)
            for label, intersection in zip(labels, labels_cnt):
                gt_id = label // OFFSET + aux_off
                pred_id = label % OFFSET + aux_off
                gt_pred_map[(gt_id, pred_id)] = intersection

        # count all matched pairs
        matched = set()
        for label_tuple, intersection in gt_pred_map.items():
            gt_label, pred_label = label_tuple
            if gt_label not in gt_segms:
                continue
            if pred_label not in pred_segms:
                continue
            if gt_label != pred_label:
                continue
            union = pred_segms[pred_label] + gt_segms[gt_label] - intersection - gt_pred_map.get((VOID, pred_label), 0)
            iou = intersection / union
            if iou > 0.5:  # Add matches from this IoU (take from original paper)
                self.pq_per_cat[gt_label].tp += 1
                self.pq_per_cat[gt_label].iou += iou
                matched.add(pred_label)

        # count false negative
        for gt_label in gt_segms:
            if gt_label in matched:
                continue
            self.pq_per_cat[gt_label].fn += 1

        # count false positives
        for pred_label, pred_area in pred_segms.items():
            if pred_label in matched:
                continue
            # intersection of the segment with VOID
            intersection = gt_pred_map.get((VOID, pred_label), 0)
            # predicted segment is ignored if more than half of the segment correspond to VOID regions
            if intersection / pred_area > 0.5:
                continue
            self.pq_per_cat[pred_label].fp += 1

    def calc_map(self, print_result: bool = False):
        """Calcule PQ-RQ-SQ maps

        Parameters
        ----------
        print_result : bool, optional
            Print results, by default False

        Returns
        -------
        Tuple[Dict,Dict]
            - Summary with all maps result (average)
            - Summary with all maps result per class
        """
        all_maps = dict()
        all_maps_per_class = dict()
        for key, cat in zip(["stuff", "thing", "all"], [False, True, None]):
            if cat is not None:
                all_maps[key], all_maps_per_class[key] = self.pq_average(cat, print_result)
            else:
                all_maps[key], all_maps_per_class[key] = self.pq_average(cat)

        if print_result:
            _print_head()
            _print_body(all_maps["all"], {})

        return all_maps, all_maps_per_class


def _print_map(average_pq: Dict, pq_per_class: Dict, suffix: str = ""):
    _print_head(suffix)
    _print_body(average_pq, pq_per_class)


def _print_head(suffix: str = ""):
    make_row = lambda vals: (" %5s |" * len(vals)) % tuple(vals)
    make_sep = lambda n: ("-------+" * (n + 1))

    print()
    print(make_sep(5))
    print(" " * 23 + "|" + make_row([v + suffix for v in ["PQ", "SQ", "RQ"]]))
    print(make_sep(5))


def _print_body(average_pq: Dict, pq_per_class: Dict):
    make_row = lambda vals: (" %5s |" * len(vals)) % tuple(vals)
    make_sep = lambda n: ("-------+" * (n + 1))

    for cat, metrics in pq_per_class.items():
        print(
            make_row(
                [
                    cat[:21] if len(cat) > 20 else cat + " " * (21 - len(cat)),
                    "%.3f" % metrics["pq"],
                    "%.3f" % metrics["sq"],
                    "%.3f" % metrics["rq"],
                ]
            )
        )
    print(make_sep(5))
    n = "%d" % average_pq["n"]
    print(
        make_row(
            [
                "total = %s" % n + " " * (13 - len(n)),
                "%.3f" % average_pq["pq"],
                "%.3f" % average_pq["sq"],
                "%.3f" % average_pq["rq"],
            ]
        )
    )
    print(make_sep(5))
