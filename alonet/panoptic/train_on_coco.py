from argparse import ArgumentParser
from alonet.panoptic import LitPanopticDetr, CocoPanoptic2Detr, PanopticHead
from alonet.detr import DetrR50Finetune, LitDetr
from alonet.common import load_training

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
    coco_loader = CocoPanoptic2Detr(
        args=args, val_stuff_ann="annotations/stuff_val2017.json", train_stuff_ann="annotations/stuff_train2017.json"
    )
    num_classes = len(coco_loader.labels_names)
    print(num_classes)

    lit_detr = load_training(
        LitDetr, 
        args=args, 
        model=DetrR50Finetune(num_classes=num_classes, aux_loss=True), 
        project_run_id="detr", 
        run_id="coco_stuff_September-10-2021-10h-32"
    )
    
    lit_panoptic = LitPanopticDetr(args, model = PanopticHead(lit_detr.model))
    lit_panoptic.run_train(
        data_loader=coco_loader, 
        args=args, 
        project="panoptic", 
        expe_name="coco_stuff"
    )


if __name__ == "__main__":
    main()
