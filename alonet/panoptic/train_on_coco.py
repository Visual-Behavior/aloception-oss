from argparse import ArgumentParser
from alonet.panoptic import CocoPanoptic2Detr
from alonet.panoptic import LitPanopticDetr

import alonet


def get_arg_parser():
    parser = ArgumentParser(conflict_handler="resolve")
    parser = alonet.common.add_argparse_args(parser)  # Common alonet parser
    parser = CocoPanoptic2Detr.add_argparse_args(parser)  # Coco detection parser
    parser = LitPanopticDetr.add_argparse_args(parser)  # LitDetr training parser
    # parser = pl.Trainer.add_argparse_args(parser) # Pytorch lightning Parser
    return parser


def main():
    """Main"""
    # Build parser
    args = get_arg_parser().parse_args()  # Parse

    # Init the Detr model with the dataset
    detr = LitPanopticDetr(args)
    coco_loader = CocoPanoptic2Detr(
        args=args, val_stuff_ann="annotations/stuff_val2017.json", train_stuff_ann="annotations/stuff_train2017.json"
    )

    detr.run_train(data_loader=coco_loader, args=args, project="detr", expe_name="detr_50")


if __name__ == "__main__":
    main()
