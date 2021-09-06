import pytorch_lightning as pl

import alodataset.transforms as T
from alonet.raft.raft_transforms import SpatialTransform, EraserTransform, ColorTransform
from alonet.common.pl_helpers import _int_or_float_type


class Data2RAFT(pl.LightningDataModule):
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.train_on_val = args.train_on_val
        self.num_workers = args.num_workers
        self.sequential = args.sequential_sampler
        self.args = args
        self.sample = args.sample if "sample" in args else False
        super().__init__()

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("Data2RAFT")
        parser.add_argument("--batch_size", type=int, default=5, help="Batch size")
        parser.add_argument("--accumulate_grad_batches", type=int, default=2, help="Accumulate gradient")
        parser.add_argument("--max_steps", type=int, default=200000)
        parser.add_argument("--train_on_val", action="store_true")
        parser.add_argument("--num_workers", type=int, default=8, help="num_workers to use on the dataset")
        parser.add_argument("--limit_val_batches", type=_int_or_float_type, default=100)
        parser.add_argument("--sequential_sampler", action="store_true", help="sample data sequentially (no shuffle)")
        return parent_parser

    def train_transform(self, frame):
        frame = ColorTransform()(frame)
        frame = EraserTransform()(frame)
        frame = SpatialTransform(crop_size=(368, 496))(frame)
        frame = T.RandomCrop(size=(368, 496))(frame)
        return frame.norm_minmax_sym()

    def val_transform(self, frame):
        return frame.norm_minmax_sym()

    def train_dataloader(self):
        raise NotImplementedError("Should be implemented in child class.")

    def val_dataloader(self):
        raise NotImplementedError("Should be implemented in child class.")

    # ----- unused pl.LightningDataModule abstract methods -----
    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass
