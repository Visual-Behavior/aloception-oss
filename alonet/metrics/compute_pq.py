# Inspired by the official panopticapi and adapted for aloception
# https://github.com/cocodataset/panopticapi/blob/master/panopticapi/evaluation.py

import numpy as np
import torch
from collections import defaultdict
from typing import Dict

import aloscene
from alodataset.utils.panoptic_utils import VOID_CLASS_ID, OFFSET
from alonet.metrics.utils import _print_body, _print_head, _print_map


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
        self.isfull = False
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
        if isthing_labels is not None:
            try:
                thing_id = isthing_labels.labels_names.index("thing")
            except:
                thing_id = len(isthing_labels.labels_names)
            self.categories.update(
                {
                    id: {"category": self.class_names[id], "isthing": it == thing_id}
                    for id, it in zip(list(cat_labels.numpy().astype(int)), list(isthing_labels.numpy().astype(int)))
                }
            )
            self.isfull = True
        else:
            self.categories.update(
                {
                    id: {"category": self.class_names[id], "isthing": True}
                    for id in list(cat_labels.numpy().astype(int))
                }
            )
            self.isfull = False

    def pq_average(self, isthing: bool = None, print_result: bool = False, **kwargs):
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
            _print_map(result, per_class_results, suffix=suffix, **kwargs)
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
        assert isinstance(p_mask.labels, aloscene.Labels) and isinstance(t_mask.labels, (dict, aloscene.Labels))

        p_mask = p_mask.to(torch.device("cpu"))
        t_mask = t_mask.to(torch.device("cpu"))

        label_set = None
        if isinstance(t_mask.labels, aloscene.Labels):
            assert hasattr(t_mask.labels, "labels_names")
            self.update_data_objects(t_mask.labels, None)
        else:
            assert "category" in t_mask.labels and hasattr(t_mask.labels["category"], "labels_names")
            if "isthing" in t_mask.labels:
                assert hasattr(t_mask.labels["isthing"], "labels_names")
                assert len(t_mask.labels["category"]) == len(t_mask.labels["isthing"])
                self.update_data_objects(t_mask.labels["category"], t_mask.labels["isthing"])
                label_set = "category"
            else:
                self.update_data_objects(t_mask.labels["category"], None)

        # Get positional ID by object
        pan_pred = p_mask.mask2id(return_cats=False) - VOID_CLASS_ID
        pred_lbl = p_mask.labels.numpy().astype("int")
        pan_gt = t_mask.mask2id(labels_set=label_set, return_cats=False) - VOID_CLASS_ID
        gt_lbl = t_mask.labels.numpy() if label_set is None else t_mask.labels[label_set].numpy()
        gt_lbl = gt_lbl.astype("int")
        VOID = 0  # VOID class in first position

        # ground truth segments area calculation
        gt_segms = {}
        labels, labels_cnt = np.unique(pan_gt, return_counts=True)
        for label, label_cnt in zip(labels, labels_cnt):
            if label == VOID:  # Ignore pixels without category
                continue
            assert gt_lbl[label - 1] < len(self.class_names)
            gt_segms[label] = {
                "area": label_cnt,  # Get area for each object
                "cat_id": gt_lbl[label - 1],  # Decode category class
            }

        # predicted segments area calculation
        pred_segms = {}
        labels, labels_cnt = np.unique(pan_pred, return_counts=True)
        for label, label_cnt in zip(labels, labels_cnt):
            if label == VOID:  # Ignore pixels without category
                continue
            assert pred_lbl[label - 1] < len(self.class_names)
            pred_segms[label] = {
                "area": label_cnt,  # Get area for each object
                "cat_id": pred_lbl[label - 1],  # Decode category class
            }

        # confusion matrix calculation if not empty views
        gt_pred_map = {}
        if len(gt_segms) > 0 and len(pred_segms) > 0:
            pan_gt_pred = pan_gt.astype(np.uint64) * OFFSET + pan_pred.astype(np.uint64)
            labels, labels_cnt = np.unique(pan_gt_pred, return_counts=True)
            for label, intersection in zip(labels, labels_cnt):
                gt_id = label // OFFSET
                pred_id = label % OFFSET
                gt_pred_map[(gt_id, pred_id)] = intersection

        # count all matched pairs
        pred_matched, gt_matched = set(), set()
        for label_tuple, intersection in gt_pred_map.items():
            gt_label, pred_label = label_tuple
            if gt_label not in gt_segms:
                continue
            if pred_label not in pred_segms:
                continue
            if gt_segms[gt_label]["cat_id"] != pred_segms[pred_label]["cat_id"]:
                continue
            union = (
                pred_segms[pred_label]["area"]
                + gt_segms[gt_label]["area"]
                - intersection
                - gt_pred_map.get((VOID, pred_label), 0)
            )
            iou = intersection / union
            if iou > 0.5:  # Add matches from this IoU (take from original paper)
                self.pq_per_cat[gt_segms[gt_label]["cat_id"]].tp += 1
                self.pq_per_cat[gt_segms[gt_label]["cat_id"]].iou += iou
                gt_matched.add(gt_label)
                pred_matched.add(pred_label)

        # count false negative
        for gt_label, gt_info in gt_segms.items():
            if gt_label in gt_matched:
                continue
            self.pq_per_cat[gt_info["cat_id"]].fn += 1

        # count false positives
        for pred_label, pred_info in pred_segms.items():
            if pred_label in pred_matched:
                continue
            # intersection of the segment with VOID
            intersection = gt_pred_map.get((VOID, pred_label), 0)
            # predicted segment is ignored if more than half of the segment correspond to VOID regions
            if intersection / pred_info["area"] > 0.5:
                continue
            self.pq_per_cat[pred_info["cat_id"]].fp += 1

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
        if self.isfull:
            keys, cats = ["stuff", "thing", "all"], [False, True, None]
        else:
            keys, cats = ["all"], [None]
        for key, cat in zip(keys, cats):
            if cat is not None or not self.isfull:
                all_maps[key], all_maps_per_class[key] = self.pq_average(
                    cat, print_result, clm_size=9, head_elm=["PQ", "SQ", "RQ"],
                )
            else:
                all_maps[key], all_maps_per_class[key] = self.pq_average(cat)

        if print_result and self.isfull:
            _print_head(head_elm=["PQ", "SQ", "RQ"], clm_size=9)
            _print_body(all_maps["all"], {})

        return all_maps, all_maps_per_class
