import random
import numpy as np
import torch

from argparse import ArgumentParser

from alonet.detr import CocoDetection2Detr
from alonet.deformable_detr import LitDeformableDetr

import alonet


def get_arg_parser():
    # Build parser
    parser = ArgumentParser(conflict_handler="resolve")
    parser = alonet.common.add_argparse_args(parser)  # Common alonet parser
    parser = CocoDetection2Detr.add_argparse_args(parser)  # Coco detection parser
    parser = LitDeformableDetr.add_argparse_args(parser)  # LitDeformableDetr training parser
    # parser = pl.Trainer.add_argparse_args(parser) # Pytorch lightning Parser
    return parser


def main():
    """Main"""
    # NOTE: for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(0)

    # torch.autograd.set_detect_anomaly(True)

    args = get_arg_parser().parse_args()  # Parse

    # Init the Detr model with the dataset
    lit_model = LitDeformableDetr(args)
    coco_loader = CocoDetection2Detr(args)

    lit_model.run_train(
        data_loader=coco_loader,
        args=args,
        project="deformable_detr_r50",
        expe_name=args.expe_name or "deformable_detr_r50",
    )


if __name__ == "__main__":
    main()
