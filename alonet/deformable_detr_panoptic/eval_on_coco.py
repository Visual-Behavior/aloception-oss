from argparse import ArgumentParser
from tqdm import tqdm
import torch

from alonet.common import add_argparse_args
from alonet.detr import CocoPanoptic2Detr
from alonet.deformable_detr_panoptic import LitPanopticDeformableDetr
from alonet.metrics import PQMetrics, ApMetrics


from aloscene import Frame


def main(args):
    """Main"""
    device = torch.device("cuda")

    # Init the DetrPanoptic model with the dataset
    coco_loader = CocoPanoptic2Detr(args, batch_size=1)
    lit_panoptic = LitPanopticDeformableDetr(args=args)
    lit_panoptic.model = lit_panoptic.model.eval().to(device)

    # Define the metric used in evaluation process
    pq_metric = PQMetrics()
    ap_metric = ApMetrics()

    # Make all predictions to use metric defined
    tbar = tqdm(total=len(coco_loader.val_dataloader()) if args.ap_limit is None else args.ap_limit)
    for it, data in enumerate(coco_loader.val_dataloader()):
        frame = Frame.batch_list(data).to(device)

        pred_boxes, pred_masks = lit_panoptic.inference(lit_panoptic(frame, threshold=0.85))
        pred_boxes, pred_masks = pred_boxes[0], pred_masks[0]
        gt_boxes = frame.boxes2d[0]  # Get gt boxes as BoundingBoxes2D.
        gt_masks = frame.segmentation[0]  # Get gt masks as Mask

        # Add samples to evaluate metrics
        pq_metric.add_sample(p_mask=pred_masks, t_mask=gt_masks)
        gt_boxes.labels, gt_masks.labels = gt_boxes.labels["category"], gt_masks.labels["category"]
        ap_metric.add_sample(p_bbox=pred_boxes, p_mask=pred_masks, t_bbox=gt_boxes, t_mask=gt_masks)

        tbar.update()
        if args.ap_limit is not None and it >= args.ap_limit:
            break

    # Show the results
    print("Total eval batch:", it)
    ap_metric.calc_map(print_result=True)
    pq_metric.calc_map(print_result=True)


if __name__ == "__main__":
    # Build parser
    parser = ArgumentParser(conflict_handler="resolve")
    parser = add_argparse_args(parser)  # Common alonet parser
    parser = CocoPanoptic2Detr.add_argparse_args(parser)  # Coco panoptic parser
    parser = LitPanopticDeformableDetr.add_argparse_args(parser)  # LitPanopticDetr training parser
    parser.add_argument(
        "--ap_limit", type=int, default=None, help="Limit AP computation at the given number of sample"
    )
    main(parser.parse_args())
