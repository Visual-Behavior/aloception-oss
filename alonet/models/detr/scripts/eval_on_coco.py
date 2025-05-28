from argparse import ArgumentParser
from dataclasses import dataclass, field
import torch

import aloscene
import alonet

from alonet.common import BaseConfig

from alonet.models.detr.data_modules import CocoDetection2Detr
from alonet.models.detr.models.detr_r50 import DetrR50
from alonet.models.detr.config.data import CocoDetectionConfig
from alonet.models.detr.config.models import DetrR50Config


@dataclass
class EvalOnCocoConfig(BaseConfig):
    model: DetrR50Config = field(default_factory=DetrR50Config)
    data: CocoDetectionConfig = field(default_factory=CocoDetectionConfig)
    ap_limit: int = field(default=None, metadata={"help": "Limit AP computation at the given number of sample"})


@torch.no_grad()
def main():
    """Main"""

    # Build parser
    parser = ArgumentParser(conflict_handler="resolve")
    parser = EvalOnCocoConfig.add_argparse_args(parser)
    args = parser.parse_args()

    # Build config
    config: EvalOnCocoConfig = EvalOnCocoConfig.from_args(args)
    config.data.batch_size = 1

    device = torch.device("cuda")

    # Init the Detr model with the dataset
    detr = DetrR50(**config.model.to_dict())
    coco_loader = CocoDetection2Detr(**config.data.to_dict())
    coco_loader.setup(stage="validation")

    detr = detr.to(device)
    detr.eval()

    ap_metrics = alonet.metrics.ApMetrics()

    for it, data in enumerate(coco_loader.val_dataloader):
        frame = aloscene.Frame.batch_list(data)
        frame = frame.to(device)

        pred_boxes = detr.inference(detr(frame))[0]
        gt_boxes = frame.boxes2d[0]

        ap_metrics.add_sample(pred_boxes, gt_boxes)

        print(f"it:{it}", end="\r")
        if config.ap_limit is not None and it > config.ap_limit:
            break

    print("Total eval batch:", it)
    ap_metrics.calc_map(print_result=True)


if __name__ == "__main__":
    main()
