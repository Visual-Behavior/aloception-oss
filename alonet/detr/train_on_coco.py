from lightning.pytorch.cli import LightningCLI
import lightning as pl
from argparse import ArgumentParser
from alonet.detr import CocoDetection2Detr
from alonet.detr import LitDetr

import alonet


def get_arg_parser():
    #parser = ArgumentParser(conflict_handler="resolve")

    # TODO set gradient_clip_val automatically to 0.1
    # check accumulate_grad_batches using pytorch ligntning
    # TODO params_update deprecated in alonet
    # TODO Setup wandb automaticly based on alonet config
    cli = LightningCLI(
        LitDetr, CocoDetection2Detr, run=False, trainer_defaults={
            "gradient_clip_val": 0.1,
            "accumulate_grad_batches": 4,
            "logger": alonet.common.get_wandb_logger(expe_name="detr_pl2", project="detr_project")
    })

    #print("parser", parser)
    #parser = alonet.common.add_argparse_args(parser, add_pl_args=False)  # Common alonet parser
    #parser = CocoDetection2Detr.add_argparse_args(parser)  # Coco detection parser
    #parser = LitDetr.add_argparse_args(parser)  # LitDetr training parser
    # parser = pl.Trainer.add_argparse_args(parser) # Pytorch lightning Parser
    #return parser


def main():
    """Main"""
    # Build parser
    args = get_arg_parser()#.parse_args()  # Parse

    # Init the Detr model with the dataset
    #detr = LitDetr(args)
    #coco_loader = CocoDetection2Detr(args)

    #detr.run_train(data_loader=coco_loader, args=args, project="detr", expe_name="detr_50")


if __name__ == "__main__":
    main()
