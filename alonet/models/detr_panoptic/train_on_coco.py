from argparse import ArgumentParser
from alonet.detr_panoptic import LitPanopticDetr
from alonet.detr import CocoPanoptic2Detr

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

    # Init the Panoptic2Detr and LitPanoptic modules
    coco_loader = CocoPanoptic2Detr(args=args)
    lit_panoptic = LitPanopticDetr(args)

    # Start training
    lit_panoptic.run_train(data_loader=coco_loader, args=args, project="detr-panoptic", expe_name="coco")


if __name__ == "__main__":
    main()
