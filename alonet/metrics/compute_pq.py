# Inspired by the official panopticapi and adapted for aloception
# https://github.com/cocodataset/panopticapi/blob/master/panopticapi/evaluation.py

import numpy as np
import torch
from collections import defaultdict
from typing import Dict, Tuple

import aloscene

VOID = -1
OFFSET = 256 * 256 * 256


class PQStatCat(object):
    def __init__(self):
        """Keep TP, FP, FN and IoU metrics per class"""
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


class PQmetrics(object):
    """Compute Panoptic, Segmentation and Recognition Qualities Metrics."""

    def __init__(self):
        self.pq_per_cat = defaultdict(PQStatCat)
        self.class_names = None

    def __getitem__(self, label_id: int):
        return self.pq_per_cat[label_id]

    def __iadd__(self, pq_stat):
        for label, pq_stat_cat in pq_stat.pq_per_cat.items():
            self.pq_per_cat[label] += pq_stat_cat
        return self

    def init_data_objects(self, class_names: list):
        """Init data objects used to compute the PQ metrics given some `class_names` list

        Parameters
        ----------
        class_names: list
            List of class_names to use to init the pq_stat_cat
        """
        self.class_names = class_names
        self.categories = {id: {"category": cname, "isthing": True} for id, cname in enumerate(class_names)}  # TODO
        self.pq_per_cat.fromkeys(range(len(class_names)))

    def pq_average(self, isthing: bool = None, print_result: bool = False) -> Tuple[Dict, Dict]:
        """Calculate SQ, RQ and PQ metrics from the categories, and thing/stuff/all if desired

        Parameters
        ----------
        categories : Dict
            Dictionary with information of if one category is 'thing' or 'stuff'
        isthing : bool
            Calculate metrics for the 'thing' category (if it True) or 'stuff' category (if it False).
            By default the procedure is executed over both
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
                cat_isthing = label_info["isthing"] == 1
                if isthing != cat_isthing:
                    continue
            iou = self.pq_per_cat[label].iou
            tp = self.pq_per_cat[label].tp
            fp = self.pq_per_cat[label].fp
            fn = self.pq_per_cat[label].fn
            if tp + fp + fn == 0:
                per_class_results[label] = {"pq": 0.0, "sq": 0.0, "rq": 0.0}
                continue
            n += 1
            pq_class = iou / (tp + 0.5 * fp + 0.5 * fn)
            sq_class = iou / tp if tp != 0 else 0
            rq_class = tp / (tp + 0.5 * fp + 0.5 * fn)
            per_class_results[label] = {"pq": pq_class, "sq": sq_class, "rq": rq_class}
            pq += pq_class
            sq += sq_class
            rq += rq_class

        result = {"pq": pq / n, "sq": sq / n, "rq": rq / n, "n": n}
        if print_result:
            self.print_map(result, per_class_results)
        return result, per_class_results

    def add_sample(self, p_mask: aloscene.Mask, t_mask: aloscene.Mask):
        """Add a new prediction and target masks to PQ metrics estimation process

        Parameters
        ----------
        p_mask : aloscene.Mask
            Predicted masks by network inference
        t_mask : aloscene.Mask
            Target masks with labels and labels_names properties

        Raises
        ------
        Exception
            p_mask and t_mask must be an aloscene.Mask object, and must have labels attribute
        """
        assert isinstance(p_mask, aloscene.Mask)
        assert isinstance(t_mask, aloscene.Mask)
        assert isinstance(p_mask.labels, aloscene.Labels)
        assert isinstance(t_mask.labels, aloscene.Labels)
        assert isinstance(p_mask.labels.scores, torch.Tensor)
        assert hasattr(t_mask.labels, "labels_names")

        p_mask = p_mask.to(torch.device("cpu"))
        t_mask = t_mask.to(torch.device("cpu"))

        if self.class_names is None:
            self.class_names = t_mask.labels.labels_names

        pan_pred = p_mask.masks2panoptic()
        pan_gt = t_mask.masks2panoptic()

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

        # confusion matrix calculation
        aux_off = VOID if VOID < 0 else 0
        pan_gt = pan_gt - aux_off
        pan_pred = pan_pred - aux_off

        pan_gt_pred = pan_gt.astype(np.uint64) * OFFSET + pan_pred.astype(np.uint64)
        gt_pred_map = {}
        labels, labels_cnt = np.unique(pan_gt_pred, return_counts=True)
        for label, intersection in zip(labels, labels_cnt):
            gt_id = label // OFFSET + aux_off
            pred_id = label % OFFSET + aux_off
            gt_pred_map[(gt_id, pred_id)] = intersection

        # count all matched pairs
        matched = set()
        for label_tuple, intersection in gt_pred_map.items():
            gt_label, pred_label = label_tuple
            if gt_label != pred_label:
                continue

            union = pred_segms[pred_label] + gt_segms[gt_label] - intersection - gt_pred_map.get((VOID, pred_label), 0)
            iou = intersection / union
            if iou > 0.5:
                self.pq_per_cat[gt_label].tp += 1
                self.pq_per_cat[gt_label].iou += iou
                matched.add(pred_label)

        # count false positives
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
        if print_result:
            print("TOTAL PQ: ")
        pq_total, _ = self.pq_average(None, print_result)
        if print_result:
            print("THINGS PQ: ")
        pq_things, _ = self.pq_average(True, print_result)
        if print_result:
            print("TOTAL PQ: ")
        pq_stuff, _ = self.pq_average(False, print_result)
        return pq_total, pq_things, pq_stuff

    @staticmethod
    def print_map(average_pq: Dict, pq_per_class: Dict):  # TODO
        print("AVERAGE PQ:")
        print(average_pq)
        print("PQ PER CLASS:")
        print(pq_per_class)
