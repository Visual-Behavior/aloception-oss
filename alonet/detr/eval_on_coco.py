from argparse import ArgumentParser
import torch

from alonet.detr import CocoDetection2Detr
from alonet.detr import LitDetr

import aloscene
import alonet


def main():
    """Main"""
    device = torch.device("cuda")

    # Build parser
    parser = ArgumentParser(conflict_handler="resolve")
    parser = alonet.common.add_argparse_args(parser)  # Common alonet parser
    parser = CocoDetection2Detr.add_argparse_args(parser)  # Coco detection parser
    parser = LitDetr.add_argparse_args(parser)  # LitDetr training parser

    parser.add_argument(
        "--ap_limit", type=int, default=None, help="Limit AP computation at the given number of sample"
    )

    args = parser.parse_args()  # Parse

    # Init the Detr model with the dataset
    detr = LitDetr(args)
    args.batch_size = 1
    coco_loader = CocoDetection2Detr(args)

    detr = detr.to(device)
    detr.model.eval()

    ap_metrics = alonet.metrics.ApMetrics()

    for it, data in enumerate(coco_loader.val_dataloader(sampler=torch.utils.data.SequentialSampler)):
        frame = aloscene.Frame.batch_list(data)
        frame = frame.to(device)

        pred_boxes = detr.inference(detr(frame))[0]
        gt_boxes = frame.boxes2d[0]

        ap_metrics.add_sample(pred_boxes, gt_boxes)

        print(f"it:{it}", end="\r")
        if args.ap_limit is not None and it > args.ap_limit:
            break

    print("Total eval batch:", it)
    ap_metrics.calc_map(print_result=True)


if __name__ == "__main__":
    main()
