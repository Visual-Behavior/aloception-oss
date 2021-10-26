import numpy as np
import torch
from matplotlib import pyplot as plt
from collections import OrderedDict

import aloscene
from aloscene import BoundingBoxes2D, BoundingBoxes3D, OrientedBoxes2D

NB_RECALL_POINTS = 101


class APDataObject:
    """Stores all the information necessary to calculate the AP for one IoU and one class."""

    def __init__(self):
        self.data_points = []
        self.areas = []
        self.num_gt_positives = 0

    def push(self, score: float, is_true: bool):
        self.data_points.append((score, is_true))

    def add_gt_positives(self, num_positives: int):
        """Call this once per image."""
        self.num_gt_positives += num_positives

    def is_empty(self) -> bool:
        return len(self.data_points) == 0 and self.num_gt_positives == 0

    def get_metrics(self) -> float:
        """Warning: result not cached."""

        if self.num_gt_positives == 0:
            return {"ap": 0, "precision": 0, "recall": 0, "precisions": [], "recalls": [], "confidences": []}

        # Sort descending by score
        self.data_points.sort(key=lambda x: -x[0])

        precisions = []
        recalls = []
        confidences = []
        num_true = 0
        num_false = 0

        # Compute the precision-recall curve. The x axis is recalls and the y axis precisions.
        for datum in self.data_points:
            # datum[1] is whether the detection a true or false positive
            if datum[1]:
                num_true += 1
            else:
                num_false += 1

            precision = num_true / (num_true + num_false)
            recall = num_true / self.num_gt_positives

            precisions.append(precision)
            recalls.append(recall)
            confidences.append(datum[0])

        precision_metric = num_true / (num_true + num_false) if (num_true + num_false) > 0 else 0
        recall_metric = num_true / self.num_gt_positives if (self.num_gt_positives) > 0 else 0

        # Smooth the curve by computing [max(precisions[i:]) for i in range(len(precisions))]
        # Basically, remove any temporary dips from the curve.
        # At least that's what I think, idk. COCOEval did it so I do too.
        for i in range(len(precisions) - 1, 0, -1):
            if precisions[i] > precisions[i - 1]:
                precisions[i - 1] = precisions[i]

        # Compute the integral of precision(recall) d_recall from recall=0->1 using fixed-length riemann summation with 101 bars.
        # print(f"use {NB_RECALL_POINTS} recall thresholds to calculate AP")
        y_range = [0] * NB_RECALL_POINTS
        x_range = np.array([x / (NB_RECALL_POINTS - 1) for x in range(NB_RECALL_POINTS)])
        recalls = np.array(recalls)
        indices = np.searchsorted(recalls, x_range, side="left")
        for bar_idx, precision_idx in enumerate(indices):
            if precision_idx < len(precisions):
                y_range[bar_idx] = precisions[precision_idx]
        x_step = 1 / (NB_RECALL_POINTS - 1)
        ap = sum(y_range) * x_step
        # y_range = [0] * 101 # idx 0 is recall == 0.0 and idx 100 is recall == 1.00
        # x_range = np.array([x / 100 for x in range(101)])
        # recalls = np.array(recalls)

        # # I realize this is weird, but all it does is find the nearest precision(x) for a given x in x_range.
        # # Basically, if the closest recall we have to 0.01 is 0.009 this sets precision(0.01) = precision(0.009).
        # # I approximate the integral this way, because that's how COCOEval does it.
        # indices = np.searchsorted(recalls, x_range, side='left')
        # for bar_idx, precision_idx in enumerate(indices):
        #     if precision_idx < len(precisions):
        #         y_range[bar_idx] = precisions[precision_idx]

        return {
            "ap": ap,
            "precision": precision_metric,
            "recall": recall_metric,
            "precisions": precisions,
            "recalls": recalls,
            "confidences": confidences,
        }


def compute_overlaps(boxes1: BoundingBoxes3D, boxes2: BoundingBoxes3D):
    """Computes IoU overlaps 3d between two sets of boxes.
    boxes1, boxes2: (N, 7) ou (M, 7)
    For better performance, pass the largest set first and the smaller second.
    """
    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    iou3d_matrix = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    iou_bev_matrix = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(iou3d_matrix.shape[1]):
        box2 = boxes2[i : i + 1].rename_(None).repeat(boxes1.shape[0], 1).reset_names()
        iou3d = box2.iou3d_with(boxes1).detach().cpu().numpy()
        iou3d_matrix[:, i] = iou3d

        iou_bev = box2.bev_boxes().rotated_iou_with(boxes1.bev_boxes()).detach().cpu().numpy()
        iou_bev_matrix[:, i] = iou_bev
    return iou3d_matrix, iou_bev_matrix


class ApMetrics3D(object):
    def __init__(self):

        self.class_names = None
        self.iou_thresholds = [x / 100.0 for x in range(50, 100, 10)]
        self.iou_thresholds_x100 = [int(x * 100) for x in self.iou_thresholds]
        self.range_breakdown = {
            "0_30": [0.0, 30.0],
            "30_50": [30.0, 50.0],
            "50_inf": [50.0, 10000.0],
        }

        self.t_bbox_3d = []
        self.t_class = []

    def init_data_objects(self, class_names):
        self.class_names = class_names
        # AP per class per IoU 3d
        self.ap_data = {
            "box": [[APDataObject() for _ in [cl for cl in self.class_names]] for _ in self.iou_thresholds]
        }
        # AP per class per IoU BEV
        self.ap_data_bev = {
            "box": [[APDataObject() for _ in [cl for cl in self.class_names]] for _ in self.iou_thresholds]
        }
        # IoU 3d 0.5
        self.ap_data_range_iou_3d_50 = {
            "box": [[APDataObject() for _ in [cl for cl in self.class_names]] for _ in self.range_breakdown]
        }
        # IoU 3d 0.7
        self.ap_data_range_iou_3d_70 = {
            "box": [[APDataObject() for _ in [cl for cl in self.class_names]] for _ in self.range_breakdown]
        }
        # IoU BEV 0.5
        self.ap_data_range_iou_bev_50 = {
            "box": [[APDataObject() for _ in [cl for cl in self.class_names]] for _ in self.range_breakdown]
        }
        # IoU BEV 0.7
        self.ap_data_range_iou_bev_70 = {
            "box": [[APDataObject() for _ in [cl for cl in self.class_names]] for _ in self.range_breakdown]
        }

    def add_dataset_sample(self, t_bbox, t_class):
        self.t_bbox_3d.append(t_bbox)
        self.t_class.append(t_class)

    def _populate_ap_objects_by_range(
        self, ap_breakdowns: dict, classes, t_bbox_range, gt_classes, iou_types, max_fp_overlap=0.01, iou_threshold=0.5
    ):

        num_pred = len(classes)
        num_gt = len(gt_classes)
        gt_used = [False] * len(gt_classes)
        pred_used = [False] * num_pred
        t_all_indices = np.array(range(gt_classes.shape[0]))

        for range_idx, range_key in enumerate(self.range_breakdown):
            for _class in set(list(classes) + list(gt_classes)):

                lower_range = self.range_breakdown[range_key][0]
                upper_range = self.range_breakdown[range_key][1]
                num_gt_for_class_range = sum(
                    [
                        1
                        for i, c in enumerate(gt_classes)
                        if (c == _class and t_bbox_range[i] >= lower_range and t_bbox_range[i] < upper_range)
                    ]
                )

                for iou_type, iou_func, score_func, indices in iou_types:

                    ap_obj = ap_breakdowns[iou_type][range_idx][_class]
                    ap_obj.add_gt_positives(num_gt_for_class_range)

                    for i in indices:
                        if pred_used[i]:
                            continue
                        else:
                            max_iou_found = iou_threshold
                            max_match_idx = -1
                            for j in range(num_gt):
                                if (
                                    gt_used[j]
                                    or t_bbox_range[j] < lower_range
                                    or t_bbox_range[j] >= upper_range
                                    or gt_classes[j] != _class
                                ):
                                    continue
                                iou = iou_func(i, j)
                                if iou > max_iou_found:
                                    max_iou_found = iou
                                    max_match_idx = j
                            if max_match_idx >= 0:
                                gt_used[max_match_idx] = True
                                pred_used[i] = True
                                ap_obj.push(score_func(i), True)
                            else:
                                # This is a false positive only if
                                # 1) This prediction does not match to any ground truth.
                                # 2) The prediction does not overlap with any other ground truth that is
                                #  not inside this breakdown.
                                gt_outside_breakdown = (t_bbox_range < lower_range) | (t_bbox_range > upper_range)
                                overlap_threshold = max_fp_overlap
                                t_indices = t_all_indices[gt_outside_breakdown]
                                max_overlap = 0
                                for j in t_indices:
                                    iou = iou_func(i, j)
                                    if iou > max_overlap:
                                        max_overlap = iou
                                    if max_overlap > overlap_threshold:
                                        break
                                if max_overlap < overlap_threshold:
                                    ap_obj.push(score_func(i), False)

    def _populate_ap_objects_all_range(self, ap_breakdowns, classes, gt_classes, iou_types):

        num_pred = len(classes)
        num_gt = len(gt_classes)
        gt_used = [False] * len(gt_classes)
        pred_used = [False] * num_pred

        for _class in set(list(classes) + list(gt_classes)):
            num_gt_for_class = sum([1 for x in gt_classes if x == _class])

            for iouIdx in range(len(self.iou_thresholds)):
                iou_threshold = self.iou_thresholds[iouIdx]
                for iou_type, iou_func, score_func, indices in iou_types:
                    gt_used = [False] * len(gt_classes)
                    pred_used = [False] * num_pred
                    ap_obj = ap_breakdowns[iou_type][iouIdx][_class]
                    ap_obj.add_gt_positives(num_gt_for_class)

                    for i in indices:
                        if pred_used[i]:
                            continue
                        else:
                            if classes[i] != _class:
                                continue
                            max_iou_found = iou_threshold
                            max_match_idx = -1
                            for j in range(num_gt):

                                if gt_used[j] or gt_classes[j] != _class:
                                    continue

                                iou = iou_func(i, j)
                                if iou > max_iou_found:
                                    max_iou_found = iou
                                    max_match_idx = j

                            if max_match_idx >= 0:
                                gt_used[max_match_idx] = True
                                ap_obj.push(score_func(i), True)
                            else:
                                ap_obj.push(score_func(i), False)

    def add_sample(
        self,
        p_bbox: BoundingBoxes3D,
        t_bbox: BoundingBoxes3D,
        p_mask: aloscene.Mask = None,
        t_mask: aloscene.Mask = None,
    ):
        assert isinstance(p_bbox, BoundingBoxes3D)
        assert isinstance(t_bbox, BoundingBoxes3D)
        assert isinstance(p_bbox.labels, aloscene.Labels)
        assert isinstance(t_bbox.labels, aloscene.Labels)
        assert isinstance(p_bbox.labels.scores, torch.Tensor)

        p_bbox = p_bbox.cuda()
        t_bbox = t_bbox.cuda()
        t_bbox_range = np.linalg.norm(t_bbox.as_tensor().cpu().numpy()[:, :3], axis=-1)
        bbox_iou_3d_cache, bbox_iou_bev_cache = compute_overlaps(p_bbox, t_bbox)

        p_labels = p_bbox.labels.cpu()
        gt_classes = t_bbox.labels.cpu()
        p_scores = p_bbox.labels.scores.cpu()
        classes = list(np.array(p_labels).astype(int))
        scores = list(np.array(p_scores).astype(float))
        self.t_class.append(gt_classes)

        if self.class_names is None:
            self.init_data_objects(gt_classes.labels_names)
        gt_classes = gt_classes.to(torch.long).numpy()

        num_pred = len(classes)
        num_gt = len(gt_classes)

        # Get box indices sorted by scores
        box_indices = sorted(range(num_pred), key=lambda i: -scores[i])

        iou_types = [
            ("box", lambda i, j: bbox_iou_3d_cache[i, j].item(), lambda i: scores[i], box_indices),  # ap
        ]

        iou_bev_types = [
            ("box", lambda i, j: bbox_iou_bev_cache[i, j].item(), lambda i: scores[i], box_indices),  # ap
        ]

        # IoU 3D
        self._populate_ap_objects_by_range(
            self.ap_data_range_iou_3d_50, classes, t_bbox_range, gt_classes, iou_types, iou_threshold=0.5
        )
        self._populate_ap_objects_by_range(
            self.ap_data_range_iou_3d_70, classes, t_bbox_range, gt_classes, iou_types, iou_threshold=0.7
        )
        self._populate_ap_objects_all_range(self.ap_data, classes, gt_classes, iou_types)
        # IoU BEV
        self._populate_ap_objects_by_range(
            self.ap_data_range_iou_bev_50, classes, t_bbox_range, gt_classes, iou_bev_types, iou_threshold=0.5
        )
        self._populate_ap_objects_by_range(
            self.ap_data_range_iou_bev_70, classes, t_bbox_range, gt_classes, iou_bev_types, iou_threshold=0.7
        )
        self._populate_ap_objects_all_range(self.ap_data_bev, classes, gt_classes, iou_bev_types)

    def calc_map(self, print_result=False, show_graph=False, export_graph=False, graph_path=None):
        def _populate_ap_all_class(ap_dict, ap_breakdowns, breakdowns):
            for b in range(len(breakdowns)):
                for _cls in range(len(self.class_names)):
                    ap_obj = ap_breakdowns["box"][b][_cls]
                    if not ap_obj.is_empty():
                        n_metrics = ap_obj.get_metrics()
                        ap_dict[b]["box"].append(n_metrics["ap"])
                        ap_dict[b]["precision"].append(n_metrics["precision"])
                        ap_dict[b]["recall"].append(n_metrics["recall"])
                        ap_dict[b]["box_ct"].append(ap_obj.num_gt_positives)

        def _populate_ap_per_class(ap_dict, ap_breakdowns, breakdowns):
            for _class in range(len(self.class_names)):
                for b in range(len(breakdowns)):
                    ap_obj = ap_breakdowns["box"][b][_class]
                    n_metrics = ap_obj.get_metrics()
                    ap_dict[self.class_names[_class]][b]["box"].append(n_metrics["ap"])
                    ap_dict[self.class_names[_class]][b]["precision"].append(n_metrics["precision"])
                    ap_dict[self.class_names[_class]][b]["recall"].append(n_metrics["recall"])
                    ap_dict[self.class_names[_class]][b]["box_ct"].append(ap_obj.num_gt_positives)

        # ============== Populate AP ==========

        # AP 3d all class, all range, iou breakdown
        aps = [{"box": [], "precision": [], "recall": [], "box_ct": []} for _ in self.iou_thresholds]
        # AP 3d per class per iou
        aps_per_class = {
            _cls: [{"box": [], "precision": [], "recall": [], "box_ct": []} for _ in self.iou_thresholds]
            for _cls in self.class_names
        }
        # AP 3d 50 all class, range breakdown
        aps50_range = [{"box": [], "precision": [], "recall": [], "box_ct": []} for _ in self.range_breakdown]
        # AP 3d 70 all class, range breakdown
        aps70_range = [{"box": [], "precision": [], "recall": [], "box_ct": []} for _ in self.range_breakdown]
        # AP 3d 50 per class per range
        aps50_class_range = {
            _cls: [{"box": [], "precision": [], "recall": [], "box_ct": []} for _ in self.range_breakdown]
            for _cls in self.class_names
        }
        # AP 3d 70 per class per range
        aps70_class_range = {
            _cls: [{"box": [], "precision": [], "recall": [], "box_ct": []} for _ in self.range_breakdown]
            for _cls in self.class_names
        }

        # AP BEV all class, all range, iou breakdown
        aps_bev = [{"box": [], "precision": [], "recall": [], "box_ct": []} for _ in self.iou_thresholds]
        # AP BEV per class per iou
        aps_per_class_bev = {
            _cls: [{"box": [], "precision": [], "recall": [], "box_ct": []} for _ in self.iou_thresholds]
            for _cls in self.class_names
        }
        # AP BEV 50 all class per range
        aps50_range_bev = [{"box": [], "precision": [], "recall": [], "box_ct": []} for _ in self.range_breakdown]
        # AP BEV 70 all class per range
        aps70_range_bev = [{"box": [], "precision": [], "recall": [], "box_ct": []} for _ in self.range_breakdown]
        # AP BEV 50 per class per range
        aps50_class_range_bev = {
            _cls: [{"box": [], "precision": [], "recall": [], "box_ct": []} for _ in self.range_breakdown]
            for _cls in self.class_names
        }
        # AP BEV 70 per class per range
        aps70_class_range_bev = {
            _cls: [{"box": [], "precision": [], "recall": [], "box_ct": []} for _ in self.range_breakdown]
            for _cls in self.class_names
        }

        # IoU 3d
        _populate_ap_all_class(aps50_range, self.ap_data_range_iou_3d_50, self.range_breakdown)
        _populate_ap_all_class(aps70_range, self.ap_data_range_iou_3d_70, self.range_breakdown)
        _populate_ap_all_class(aps, self.ap_data, self.iou_thresholds)
        _populate_ap_per_class(aps_per_class, self.ap_data, self.iou_thresholds)
        _populate_ap_per_class(aps50_class_range, self.ap_data_range_iou_3d_50, self.range_breakdown)
        _populate_ap_per_class(aps70_class_range, self.ap_data_range_iou_3d_70, self.range_breakdown)

        # IoU BEV
        _populate_ap_all_class(aps50_range_bev, self.ap_data_range_iou_bev_50, self.range_breakdown)
        _populate_ap_all_class(aps70_range_bev, self.ap_data_range_iou_bev_70, self.range_breakdown)
        _populate_ap_all_class(aps_bev, self.ap_data_bev, self.iou_thresholds)
        _populate_ap_per_class(aps_per_class_bev, self.ap_data_bev, self.iou_thresholds)
        _populate_ap_per_class(aps50_class_range_bev, self.ap_data_range_iou_bev_50, self.range_breakdown)
        _populate_ap_per_class(aps70_class_range_bev, self.ap_data_range_iou_bev_70, self.range_breakdown)

        # =============== Draw graph: IoU 3d 50 ==========
        cross_class_ap50_obj = APDataObject()
        per_class_ap50_metrics = {}
        for _class in range(len(self.class_names)):
            for iou_idx in range(len(self.iou_thresholds)):
                ap_obj = self.ap_data["box"][iou_idx][_class]
                if not ap_obj.is_empty():
                    n_metrics = ap_obj.get_metrics()
                    if iou_idx == 0:
                        per_class_ap50_metrics[self.class_names[_class]] = n_metrics
                        cross_class_ap50_obj.add_gt_positives(ap_obj.num_gt_positives)
                        for score, true_false in ap_obj.data_points:
                            cross_class_ap50_obj.push(score, true_false)
        cross_clas_ap50_metrics = cross_class_ap50_obj.get_metrics()

        # Precision/Confidence curve
        plt.figure(figsize=(20, 5))
        plt.subplot(131)
        plt.plot(cross_clas_ap50_metrics["confidences"], cross_clas_ap50_metrics["precisions"], label="all")
        for _class in per_class_ap50_metrics:
            plt.plot(
                per_class_ap50_metrics[_class]["confidences"],
                per_class_ap50_metrics[_class]["precisions"],
                label=_class,
            )
        plt.xlabel("Confidence")
        plt.ylabel("Precision")
        plt.legend()
        plt.title("Precision/confidence curve")

        # Recall/Confidence curve
        plt.subplot(132)
        plt.plot(cross_clas_ap50_metrics["confidences"], cross_clas_ap50_metrics["recalls"], label="all")
        for _class in per_class_ap50_metrics:
            plt.plot(
                per_class_ap50_metrics[_class]["confidences"], per_class_ap50_metrics[_class]["recalls"], label=_class
            )
        plt.xlabel("Confidence")
        plt.ylabel("Recall")
        plt.legend()
        plt.title("Recall/confidence curve")

        # Roc curve: Recall vs Precision curve
        plt.subplot(133)
        plt.plot(cross_clas_ap50_metrics["recalls"], cross_clas_ap50_metrics["precisions"], label="all")
        for _class in per_class_ap50_metrics:
            plt.plot(
                per_class_ap50_metrics[_class]["recalls"], per_class_ap50_metrics[_class]["precisions"], label=_class
            )
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        plt.title("ROC curve: Recall vs Precision")
        plt.tight_layout()
        if graph_path is not None and export_graph:
            plt.savefig(graph_path)

        # ============= Calculate mAP ===============
        all_maps = {"box": OrderedDict(), "box_ct": OrderedDict(), "precision": OrderedDict(), "recall": OrderedDict()}
        per_class_all_maps = {
            _cls: {"box": OrderedDict(), "box_ct": OrderedDict(), "precision": OrderedDict(), "recall": OrderedDict()}
            for _cls in self.class_names
        }
        per_class_range_maps_50 = {
            _cls: {"box": OrderedDict(), "box_ct": OrderedDict(), "precision": OrderedDict(), "recall": OrderedDict()}
            for _cls in self.class_names
        }
        per_class_range_maps_70 = {
            _cls: {"box": OrderedDict(), "box_ct": OrderedDict(), "precision": OrderedDict(), "recall": OrderedDict()}
            for _cls in self.class_names
        }
        all_maps_per_range = {
            "box": OrderedDict(),
            "box_ct": OrderedDict(),
            "precision": OrderedDict(),
            "recall": OrderedDict(),
        }
        all_maps_per_range_iou_70 = {
            "box": OrderedDict(),
            "box_ct": OrderedDict(),
            "precision": OrderedDict(),
            "recall": OrderedDict(),
        }

        all_maps_bev = {
            "box": OrderedDict(),
            "box_ct": OrderedDict(),
            "precision": OrderedDict(),
            "recall": OrderedDict(),
        }
        per_class_all_maps_bev = {
            _cls: {"box": OrderedDict(), "box_ct": OrderedDict(), "precision": OrderedDict(), "recall": OrderedDict()}
            for _cls in self.class_names
        }
        per_class_range_maps_50_bev = {
            _cls: {"box": OrderedDict(), "box_ct": OrderedDict(), "precision": OrderedDict(), "recall": OrderedDict()}
            for _cls in self.class_names
        }
        per_class_range_maps_70_bev = {
            _cls: {"box": OrderedDict(), "box_ct": OrderedDict(), "precision": OrderedDict(), "recall": OrderedDict()}
            for _cls in self.class_names
        }
        all_maps_per_range_bev = {
            "box": OrderedDict(),
            "box_ct": OrderedDict(),
            "precision": OrderedDict(),
            "recall": OrderedDict(),
        }
        all_maps_per_range_iou_70_bev = {
            "box": OrderedDict(),
            "box_ct": OrderedDict(),
            "precision": OrderedDict(),
            "recall": OrderedDict(),
        }

        def _calculate_mAP_all_class(map_dict: OrderedDict, ap_dict, breakdowns):
            for iou_type in ("box", "precision", "recall", "box_ct"):
                map_dict[iou_type]["all"] = 0  # Make this first in the ordereddict
                for i, b in enumerate(breakdowns):
                    if iou_type != "box_ct":
                        mAP = (
                            sum(ap_dict[i][iou_type]) / len(ap_dict[i][iou_type]) * 100
                            if len(ap_dict[i][iou_type]) > 0
                            else 0
                        )
                        map_dict[iou_type][b] = mAP
                    else:
                        mAP = sum(ap_dict[i][iou_type]) if len(ap_dict[i][iou_type]) > 0 else 0
                        map_dict[iou_type][b] = mAP
                map_dict[iou_type]["all"] = sum(map_dict[iou_type].values()) / (len(map_dict[iou_type].values()) - 1)

        def _calculate_mAP_per_class(map_dict: OrderedDict, ap_dict, breakdowns):
            for iou_type in ("box", "precision", "recall", "box_ct"):
                for _cls in self.class_names:
                    map_dict[_cls][iou_type]["all"] = 0  # Make this first in the ordereddict
                    for i, b in enumerate(breakdowns):
                        if iou_type != "box_ct":
                            mAP = (
                                sum(ap_dict[_cls][i][iou_type]) / len(ap_dict[_cls][i][iou_type]) * 100
                                if len(ap_dict[_cls][i][iou_type]) > 0
                                else 0
                            )
                            map_dict[_cls][iou_type][str(b)] = mAP
                        else:
                            mAP = sum(ap_dict[_cls][i][iou_type]) if len(ap_dict[_cls][i][iou_type]) > 0 else 0
                            map_dict[_cls][iou_type][str(b)] = mAP
                    map_dict[_cls][iou_type]["all"] = sum(map_dict[_cls][iou_type].values()) / (
                        len(map_dict[_cls][iou_type].values()) - 1 + 1e-3
                    )  # avoid Divide by 0

        # IoU 3d
        _calculate_mAP_all_class(all_maps, aps, self.iou_thresholds_x100)
        _calculate_mAP_all_class(all_maps_per_range, aps50_range, self.range_breakdown)  # IoU 3d 50
        _calculate_mAP_all_class(all_maps_per_range_iou_70, aps70_range, self.range_breakdown)  # IoU 3d 70
        _calculate_mAP_per_class(per_class_range_maps_50, aps50_class_range, self.range_breakdown)
        _calculate_mAP_per_class(per_class_range_maps_70, aps70_class_range, self.range_breakdown)
        # IoU BEV
        _calculate_mAP_all_class(all_maps_bev, aps_bev, self.iou_thresholds_x100)
        _calculate_mAP_all_class(all_maps_per_range_bev, aps50_range_bev, self.range_breakdown)  # IoU 3d 50
        _calculate_mAP_all_class(all_maps_per_range_iou_70_bev, aps70_range_bev, self.range_breakdown)  # IoU 3d 70
        _calculate_mAP_per_class(per_class_range_maps_50_bev, aps50_class_range_bev, self.range_breakdown)
        _calculate_mAP_per_class(per_class_range_maps_70_bev, aps70_class_range_bev, self.range_breakdown)

        if print_result:
            print("=" * 30)
            print("AP 3D (and AP BEV if shown)")
            print("=========== All class - All distance ===========")
            print_maps(all_maps, all_maps_bev)

            print("=========== IoU 0.5 ===========")
            print("All class, Range:")
            print_maps(all_maps_per_range, all_maps_per_range_bev)
            for _cls in self.class_names:
                print_maps(per_class_range_maps_50[_cls], per_class_range_maps_50_bev[_cls], name=_cls)

            print("=========== IoU 0.7 ===========")
            print("All class, Range:")
            print_maps(all_maps_per_range_iou_70, all_maps_per_range_iou_70_bev)
            for _cls in self.class_names:
                print_maps(per_class_range_maps_70[_cls], per_class_range_maps_70_bev[_cls], name=_cls)

        return all_maps, per_class_all_maps, all_maps_per_range, cross_clas_ap50_metrics, per_class_ap50_metrics


def print_maps(all_maps, all_maps_bev=None, name=""):
    # Warning: hacky
    if all_maps_bev is None:
        make_row = lambda vals: (" %5s |" * len(vals)) % tuple(vals)
        make_sep = lambda n: ("--------" * (n + 1))
    else:
        make_row = lambda vals: (" %12s |" * len(vals)) % tuple(vals)
        make_sep = lambda n: ("--------------" * (n + 1))

    # Print header
    if len(name) > 0:
        print(name)
        print(make_sep(len(all_maps["box"]) + 1))
    print(make_row(["    \t\t"] + [(".%d " % x if isinstance(x, int) else x + " ") for x in all_maps["box"].keys()]))
    print(make_sep(len(all_maps["box"]) + 1))

    # Print data
    if all_maps_bev is None:
        for iou_type in ("box", "precision", "recall"):
            print(
                make_row(
                    [f"{iou_type}\t" if iou_type == "precision" else f"{iou_type}\t\t"]
                    + [f"{x:.2f}" for x in all_maps[iou_type].values()]
                )
            )
    else:
        for iou_type in ("box", "precision", "recall"):
            print(
                make_row(
                    [f"{iou_type}\t\t" if iou_type == "precision" else f"{iou_type}\t\t"]
                    + [
                        f"{x:.2f}/{y:.2f}"
                        for x, y in zip(all_maps[iou_type].values(), all_maps_bev[iou_type].values())
                    ]
                )
            )

    iou_type = "box_ct"
    print(make_row([f"{iou_type}\t\t"] + [f"{x:.0f}" for x in all_maps[iou_type].values()]))

    print(make_sep(len(all_maps["box"]) + 1))
    print()
