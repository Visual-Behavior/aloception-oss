from argparse import ArgumentParser
from alonet.detr import CocoDetection2Detr
from alonet.detr import LitDetr

import alonet


def get_arg_parser():
    parser = ArgumentParser(conflict_handler="resolve")
    parser = alonet.common.add_argparse_args(parser)  # Common alonet parser
    parser = CocoDetection2Detr.add_argparse_args(parser)  # Coco detection parser
    parser.add_argument(
        "--use_sample", action="store_true", help="Download a sample for train process (Default: %(default)s)"
    )
    parser = LitDetr.add_argparse_args(parser)  # LitDetr training parser
    # parser = pl.Trainer.add_argparse_args(parser) # Pytorch lightning Parser
    return parser


def main():
    """Main"""
    # Build parser
    args = get_arg_parser().parse_args()  # Parse

    # Init the Detr model with the dataset
    detr = LitDetr(args)
    coco_loader = CocoDetection2Detr(args, sample=args.use_sample)

    detr.run_train(data_loader=coco_loader, args=args, project="detr", expe_name="detr_50")


if __name__ == "__main__":
    main()
