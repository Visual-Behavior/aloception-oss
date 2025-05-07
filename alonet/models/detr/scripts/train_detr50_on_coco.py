from argparse import ArgumentParser, Namespace
from dataclasses import field, dataclass
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from alonet.common import BaseConfig, get_rank

from alonet.models.detr.data_modules.coco_detection2detr import CocoDetection2Detr
from alonet.models.detr.trainer import DetrTrainer

from alonet.models.detr.config.data.coco_detection import CocoDetectionConfig
from alonet.models.detr.config.models.detr_r50 import DetrR50Config
from alonet.models.detr.config.training.detr_training import DetrTrainingConfig


@dataclass
class Detr50CocoTrainingConfig(BaseConfig):
    data_module: CocoDetectionConfig = field(default_factory=CocoDetectionConfig)
    model: DetrR50Config = field(default_factory=DetrR50Config)
    trainer: DetrTrainingConfig = field(default_factory=DetrTrainingConfig)


def get_arg_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser = Detr50CocoTrainingConfig.add_argparse_args(parser)
    parser.add_argument("--compile", action="store_true", help="Compile the model")
    parser.add_argument("--save_config", type=str, default=None, help="Save the config")
    return parser


def main(args: Namespace):
    if args.save_config is not None:
        # Initialize training config
        training_config: Detr50CocoTrainingConfig = Detr50CocoTrainingConfig.from_args(args)
        training_config.save(args.save_config)
        return

    with_ddp = torch.cuda.device_count() > 1
    if with_ddp:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl")

    # Initialize training config
    training_config: Detr50CocoTrainingConfig = Detr50CocoTrainingConfig.from_args(args)

    # Initialize data module
    data_module = CocoDetection2Detr(**training_config.data_module.to_dict())
    data_module.setup("training")

    # Initialize trainer
    trainer = DetrTrainer(**training_config.trainer.to_dict())

    if with_ddp:
        trainer.model = DDP(trainer.model, device_ids=[get_rank()])

    if args.compile:
        trainer.model = torch.compile(trainer.model)

    # Train
    trainer.train(data_module.train_dataloader, data_module.val_dataloader)


if __name__ == "__main__":
    args = get_arg_parser().parse_args()
    main(args)
