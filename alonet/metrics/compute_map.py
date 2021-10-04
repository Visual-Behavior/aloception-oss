# import matplotlib.pyplot as plt
import numpy as np
import torch

from collections import OrderedDict
import aloscene


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
        # self.areas = areas

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

        # Compute the integral of precision(recall) d_recall from recall=0->1 using fixed-length riemann summation
        # with 101 bars.
        y_range = [0] * 101  # idx 0 is recall == 0.0 and idx 100 is recall == 1.00
        x_range = np.array([x / 100 for x in range(101)])
        recalls = np.array(recalls)

        # I realize this is weird, but all it does is find the nearest precision(x) for a given x in x_range.
        # Basically, if the closest recall we have to 0.01 is 0.009 this sets precision(0.01) = precision(0.009).
        # I approximate the integral this way, because that's how COCOEval does it.
        indices = np.searchsorted(recalls, x_range, side="left")
        for bar_idx, precision_idx in enumerate(indices):
            if precision_idx < len(precisions):
                y_range[bar_idx] = precisions[precision_idx]

        return {
            "ap": sum(y_range) / len(y_range),
            "precision": precision_metric,
            "recall": recall_metric,
            "precisions": precisions,
            "recalls": recalls,
            "confidences": confidences,
        }

        # Finally compute the riemann sum to get our integral.
        # avg([precision(x) for x in 0:0.01:1])
        return


class ApMetrics(object):
    def __init__(self, iou_thresholds=[x / 100.0 for x in range(50, 100, 5)], compute_per_size_ap=False):
        """Compute AP Metrics.

        Parameters
        ----------
        iou_thresholds: list
            List of thresholds to use. Default: [0.5, 0.55, .... 0.9, 0.95]
        compute_per_size_ap: bool
            False by default. If True, will compute the IOU per boxes area/size.
        """
        self.iou_thresholds = iou_thresholds
        self.class_names = None
        self.compute_per_size_ap = compute_per_size_ap

    def init_data_objects(self, class_names: list):
        """Init data objects used to compute the AP given some `class_names` list

        Parameters
        ----------
        class_names: list
            List of class_names to use to Init the Data objects
        """
        self.class_names = class_names

        self.ap_data = {
            "box": [[APDataObject() for _ in [cl for cl in class_names]] for _ in self.iou_thresholds],
            "mask": [[APDataObject() for _ in [cl for cl in class_names]] for _ in self.iou_thresholds],
        }

        if self.compute_per_size_ap:
            self.objects_sizes = {
                "size_0": [0.0, 0.00125],
                "size_1": [0.00125, 0.0025],
                "size_2": [0.0025, 0.005],
                "size_3": [0.005, 0.01],
                "size_4": [0.01, 0.02],
                "size_5": [0.02, 0.04],
                "size_6": [0.04, 0.08],
                "size_7": [0.08, 0.16],
                "size_8": [0.16, 0.32],
            }
            self.ap_data_size = {
                "box": [[APDataObject() for _ in [cl for cl in class_names]] for _ in self.objects_sizes],
                "mask": [[APDataObject() for _ in [cl for cl in class_names]] for _ in self.objects_sizes],
            }

    def add_sample(
        self,
        p_bbox: aloscene.BoundingBoxes2D,
        t_bbox: aloscene.BoundingBoxes2D,
        p_mask: aloscene.Mask = None,
        t_mask: aloscene.Mask = None,
    ):
        """Add sample to compute the AP.

        Parameters
        ----------
        p_bbox: `aloscene.BoundingBoxes2D`
            predicted boxes
        t_bbox: `aloscene.BoundingBoxes2D`
            Target boxes with `aloscene.labels` with the `labels_names` property set.
        p_mask: `aloscene.Mask`
            Apply APmask metric
        t_mask: `aloscene.Mask`
            Apply APmask metric
        """
        assert isinstance(p_bbox, aloscene.BoundingBoxes2D)
        assert isinstance(t_bbox, aloscene.BoundingBoxes2D)
        assert isinstance(p_bbox.labels, aloscene.Labels)
        assert isinstance(t_bbox.labels, aloscene.Labels)
        assert isinstance(p_bbox.labels.scores, torch.Tensor)
        assert isinstance(p_mask, (type(None), aloscene.Mask))
        assert isinstance(t_mask, (type(None), aloscene.Mask))

        p_bbox = p_bbox.to(torch.device("cpu"))
        p_labels = p_bbox.labels
        p_scores = p_bbox.labels.scores
        t_bbox = t_bbox.to(torch.device("cpu"))
        gt_classes = t_bbox.labels
        p_mask = None if p_mask is None else p_mask.to(torch.device("cpu"))
        t_mask = None if t_mask is None else t_mask.to(torch.device("cpu"))

        if self.class_names is None:
            self.init_data_objects(gt_classes.labels_names)

        num_crowd = 0

        classes = list(np.array(p_labels).astype(int))
        gt_classes = list(np.array(gt_classes).astype(int))
        scores = list(np.array(p_scores).astype(float))

        box_scores = scores
        mask_scores = scores

        num_pred = len(classes)
        num_gt = len(gt_classes)

        bbox_iou_cache = np.array(p_bbox.iou_with(t_bbox))  # compute_overlaps(p_bbox, t_bbox)

        # Split the AP in different size:
        t_bbox_area = list(np.array(t_bbox.rel_area()))
        p_bbox_area = list(np.array(p_bbox.rel_area()))

        if p_mask is not None and t_mask is not None:
            mask_iou_cache = np.array(p_mask.iou_with(t_mask))
        else:
            mask_iou_cache = np.zeros((p_bbox.shape[0], t_bbox.shape[0]))

        crowd_mask_iou_cache = None
        crowd_bbox_iou_cache = None

        box_indices = sorted(range(num_pred), key=lambda i: -box_scores[i])
        mask_indices = sorted(box_indices, key=lambda i: -mask_scores[i])

        iou_types = [
            (
                "box",
                lambda i, j: bbox_iou_cache[i, j].item(),
                lambda i, j: crowd_bbox_iou_cache[i, j].item(),
                lambda i: box_scores[i],
                box_indices,
            ),
            (
                "mask",
                lambda i, j: mask_iou_cache[i, j].item(),
                lambda i, j: crowd_mask_iou_cache[i, j].item(),
                lambda i: mask_scores[i],
                mask_indices,
            ),
        ]

        if self.compute_per_size_ap:
            for size_idx, size_key in enumerate(self.objects_sizes):

                for _class in set(list(classes) + list(gt_classes)):
                    lower_size = self.objects_sizes[size_key][0]
                    upper_size = self.objects_sizes[size_key][1]

                    num_gt_for_class_size = sum(
                        [
                            1
                            for i, c in enumerate(gt_classes)
                            if (c == _class and t_bbox_area[i] >= lower_size and t_bbox_area[i] < upper_size)
                        ]
                    )

                    for iou_type, iou_func, crowd_func, score_func, indices in iou_types:

                        gt_used = [False] * len(gt_classes)

                        ap_obj = self.ap_data_size[iou_type][size_idx][_class]
                        ap_obj.add_gt_positives(num_gt_for_class_size)

                        for i in indices:
                            max_iou_found = 0.5
                            max_match_idx = -1
                            for j in range(num_gt):
                                if (
                                    gt_used[j]
                                    or t_bbox_area[j] < lower_size
                                    or t_bbox_area[j] >= upper_size
                                    or gt_classes[j] != _class
                                ):
                                    continue

                            if max_match_idx >= 0:
                                gt_used[max_match_idx] = True
                                ap_obj.push(score_func(i), True)
                            else:
                                # If the detection matches a crowd, we can just ignore it
                                matched_crowd = False
                                if num_crowd > 0:
                                    for j in range(len(crowd_classes)):
                                        iou = crowd_func(i, j)
                                        if iou > 0.5:
                                            matched_crowd = True
                                            break
                                # All this crowd code so that we can make sure that our eval code gives the
                                # same result as COCOEval. There aren't even that many crowd annotations to
                                # begin with, but accuracy is of the utmost importance.
                                if not matched_crowd and p_bbox_area[i] >= lower_size and p_bbox_area[i] < upper_size:
                                    ap_obj.push(score_func(i), False)

        for _class in set(list(classes) + list(gt_classes)):
            num_gt_for_class = sum([1 for x in gt_classes if x == _class])

            for iouIdx in range(len(self.iou_thresholds)):
                iou_threshold = self.iou_thresholds[iouIdx]
                for iou_type, iou_func, crowd_func, score_func, indices in iou_types:
                    gt_used = [False] * len(gt_classes)

                    ap_obj = self.ap_data[iou_type][iouIdx][_class]
                    ap_obj.add_gt_positives(num_gt_for_class)

                    for i in indices:
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
                            # If the detection matches a crowd, we can just ignore it
                            matched_crowd = False

                            if num_crowd > 0:
                                for j in range(len(crowd_classes)):
                                    if crowd_classes[j] != _class:
                                        continue

                                    iou = crowd_func(i, j)

                                    if iou > iou_threshold:
                                        matched_crowd = True
                                        break

                            # All this crowd code so that we can make sure that our eval code gives the
                            # same result as COCOEval. There aren't even that many crowd annotations to
                            # begin with, but accuracy is of the utmost importance.
                            if not matched_crowd:
                                ap_obj.push(score_func(i), False)

    def calc_map(self, print_result=False):

        # AP across class
        aps = [{"box": [], "precision": [], "recall": [], "mask": [], "box_ct": []} for _ in self.iou_thresholds]
        # AP 50 per size
        if self.compute_per_size_ap:
            aps50_size = [
                {"box": [], "precision": [], "recall": [], "mask": [], "box_ct": []} for _ in self.objects_sizes
            ]
        # AP per class
        aps_per_class = {
            _cls: [{"box": [], "precision": [], "recall": [], "mask": [], "box_ct": []} for _ in self.iou_thresholds]
            for _cls in self.class_names
        }

        if self.compute_per_size_ap:
            for iou_type in ("box", "mask"):
                for size_idx in range(len(self.objects_sizes)):
                    for _class in range(len(self.class_names)):
                        ap_obj = self.ap_data_size[iou_type][size_idx][_class]
                        if not ap_obj.is_empty():
                            n_metrics = ap_obj.get_metrics()
                            # if n_metrics is not None:
                            aps50_size[size_idx][iou_type].append(n_metrics["ap"])
                            if iou_type == "box":
                                aps50_size[size_idx]["precision"].append(n_metrics["precision"])
                                aps50_size[size_idx]["recall"].append(n_metrics["recall"])
                                # print('size_idx', size_idx, iou_type, size_idx, _class,  ap_obj.num_gt_positives)
                                aps50_size[size_idx]["box_ct"].append(ap_obj.num_gt_positives)

        cross_class_ap50_obj = APDataObject()
        per_class_ap50_metrics = {}

        for _class in range(len(self.class_names)):
            for iou_idx in range(len(self.iou_thresholds)):
                for iou_type in ("box", "mask"):
                    ap_obj = self.ap_data[iou_type][iou_idx][_class]
                    # print("_class", _class, ap_obj.is_empty(), iou_type, iou_idx)
                    if not ap_obj.is_empty():

                        n_metrics = ap_obj.get_metrics()

                        if iou_idx == 0 and iou_type == "box":
                            per_class_ap50_metrics[self.class_names[_class]] = n_metrics

                            cross_class_ap50_obj.add_gt_positives(ap_obj.num_gt_positives)
                            for score, true_false in ap_obj.data_points:
                                cross_class_ap50_obj.push(score, true_false)

                        # TODO: Get the precisions and recall along with the box and mask
                        # if n_metrics is not None:
                        aps_per_class[self.class_names[_class]][iou_idx][iou_type].append(n_metrics["ap"])
                        aps[iou_idx][iou_type].append(n_metrics["ap"])

                        if iou_type == "box":
                            aps_per_class[self.class_names[_class]][iou_idx]["precision"].append(
                                n_metrics["precision"]
                            )
                            aps[iou_idx]["precision"].append(n_metrics["precision"])
                            aps_per_class[self.class_names[_class]][iou_idx]["recall"].append(n_metrics["recall"])
                            aps[iou_idx]["recall"].append(n_metrics["recall"])
                            aps_per_class[self.class_names[_class]][iou_idx]["box_ct"].append(ap_obj.num_gt_positives)
                            aps[iou_idx]["box_ct"].append(ap_obj.num_gt_positives)

        cross_clas_ap50_metrics = cross_class_ap50_obj.get_metrics()

        """
        # PLOT TODO
        # Precision/Confidence curve
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
        plt.show()

        # PLOT TODO
        # Recall/Confidence curve
        plt.plot(cross_clas_ap50_metrics["confidences"], cross_clas_ap50_metrics["recalls"], label="all")
        for _class in per_class_ap50_metrics:
            plt.plot(
                per_class_ap50_metrics[_class]["confidences"], per_class_ap50_metrics[_class]["recalls"], label=_class
            )
        plt.xlabel("Confidence")
        plt.ylabel("Recall")
        plt.legend()
        plt.title("Recall/confidence curve")
        plt.show()

        # PLOT TODO
        # Roc curve: Recall vs Confidence curve
        plt.plot(cross_clas_ap50_metrics["recalls"], cross_clas_ap50_metrics["precisions"], label="all")
        for _class in per_class_ap50_metrics:
            plt.plot(
                per_class_ap50_metrics[_class]["recalls"], per_class_ap50_metrics[_class]["precisions"], label=_class
            )
        plt.xlabel("Recall")
        plt.ylabel("Confidence")
        plt.legend()
        plt.title("ROC curve: Recall vs Precision")
        plt.show()
        """

        all_maps = {
            "box": OrderedDict(),
            "box_ct": OrderedDict(),
            "mask": OrderedDict(),
            "precision": OrderedDict(),
            "recall": OrderedDict(),
        }
        per_class_all_maps = {
            _cls: {
                "box": OrderedDict(),
                "box_ct": OrderedDict(),
                "mask": OrderedDict(),
                "precision": OrderedDict(),
                "recall": OrderedDict(),
            }
            for _cls in self.class_names
        }
        all_maps_per_size = {
            "box": OrderedDict(),
            "box_ct": OrderedDict(),
            "mask": OrderedDict(),
            "precision": OrderedDict(),
            "recall": OrderedDict(),
        }

        for iou_type in ("box", "mask", "precision", "recall", "box_ct"):

            all_maps[iou_type]["all"] = 0  # Make this first in the ordereddict
            for i, threshold in enumerate(self.iou_thresholds):

                if iou_type != "box_ct":
                    mAP = sum(aps[i][iou_type]) / len(aps[i][iou_type]) * 100 if len(aps[i][iou_type]) > 0 else 0
                    all_maps[iou_type][int(threshold * 100)] = mAP
                else:
                    mAP = sum(aps[i][iou_type]) if len(aps[i][iou_type]) > 0 else 0
                    all_maps[iou_type][int(threshold * 100)] = mAP

            all_maps[iou_type]["all"] = sum(all_maps[iou_type].values()) / (len(all_maps[iou_type].values()) - 1)

            if self.compute_per_size_ap:
                all_maps_per_size[iou_type]["all"] = 0  # Make this first in the ordereddict
                for i, size in enumerate(self.objects_sizes):
                    if iou_type != "box_ct":
                        mAP = (
                            sum(aps50_size[i][iou_type]) / len(aps50_size[i][iou_type]) * 100
                            if len(aps50_size[i][iou_type]) > 0
                            else 0
                        )
                        all_maps_per_size[iou_type][str(size)] = mAP
                    else:
                        # print('aps50_size[i][iou_type]', aps50_size[i][iou_type], i, iou_type)
                        mAP = sum(aps50_size[i][iou_type]) if len(aps50_size[i][iou_type]) > 0 else 0
                        all_maps_per_size[iou_type][str(size)] = mAP
                all_maps_per_size[iou_type]["all"] = sum(all_maps_per_size[iou_type].values()) / (
                    len(all_maps_per_size[iou_type].values()) - 1
                )

            for _cls in self.class_names:
                per_class_all_maps[_cls][iou_type]["all"] = 0  # Make this first in the ordereddict
                for i, threshold in enumerate(self.iou_thresholds):
                    if iou_type != "box_ct":
                        mAP = (
                            sum(aps_per_class[_cls][i][iou_type]) / len(aps_per_class[_cls][i][iou_type]) * 100
                            if len(aps_per_class[_cls][i][iou_type]) > 0
                            else 0
                        )
                        per_class_all_maps[_cls][iou_type][int(threshold * 100)] = mAP
                    else:
                        mAP = sum(aps_per_class[_cls][i][iou_type]) if len(aps_per_class[_cls][i][iou_type]) > 0 else 0
                        per_class_all_maps[_cls][iou_type][int(threshold * 100)] = mAP

                per_class_all_maps[_cls][iou_type]["all"] = sum(per_class_all_maps[_cls][iou_type].values()) / (
                    len(per_class_all_maps[_cls][iou_type].values()) - 1
                )

        if print_result:
            print("AP50,Size:")
            if self.compute_per_size_ap:
                print_maps(all_maps_per_size)
            print("All")
            print_maps(all_maps)

            # Per class bbox map50
            per_class_data = []
            for _cls in self.class_names:
                per_class_data.append(
                    [
                        _cls,
                        round(per_class_all_maps[_cls]["box"]["all"], 2),
                        round(per_class_all_maps[_cls]["box"][50], 2),
                        round(per_class_all_maps[_cls]["box"][70], 2),
                        round(per_class_all_maps[_cls]["precision"]["all"], 2),
                        round(per_class_all_maps[_cls]["precision"][50], 2),
                        round(per_class_all_maps[_cls]["precision"][70], 2),
                        round(per_class_all_maps[_cls]["recall"]["all"], 2),
                        round(per_class_all_maps[_cls]["recall"][50], 2),
                        round(per_class_all_maps[_cls]["recall"][70], 2),
                    ]
                )

            print(
                "\t".join(
                    [
                        v.ljust(10)
                        for v in [
                            "class name",
                            "AP",
                            "AP50",
                            "AP70",
                            "precision",
                            "precision50",
                            "precision70",
                            "recall",
                            "recall50",
                            "recall70",
                        ]
                    ]
                )
            )
            for data in per_class_data:
                print(
                    "\t".join(
                        [str(v).ljust(10) if len(str(v)) < 10 else "{}...".format(str(v)[:7]).ljust(10) for v in data]
                    )
                )

        return all_maps, per_class_all_maps, all_maps_per_size, cross_clas_ap50_metrics, per_class_ap50_metrics


def print_maps(all_maps, name=""):
    # Warning: hacky
    make_row = lambda vals: (" %5s |" * len(vals)) % tuple(vals)
    make_sep = lambda n: ("-------+" * (n + 1))

    if len(name) > 0:
        print(name)
        print(make_sep(len(all_maps["box"]) + 1))
    print(make_row(["    \t\t"] + [(".%d " % x if isinstance(x, int) else x + " ") for x in all_maps["box"].keys()]))
    print(make_sep(len(all_maps["box"]) + 1))
    for iou_type in ("box", "mask", "precision", "recall", "box_ct"):
        print(
            make_row(
                ["{}\t".format(iou_type) if iou_type == "precision" else "{}\t\t".format(iou_type)]
                + ["%.2f" % x if x < 100 else "%.1f" % x for x in all_maps[iou_type].values()]
            )
        )
    print(make_sep(len(all_maps["box"]) + 1))
    print()
