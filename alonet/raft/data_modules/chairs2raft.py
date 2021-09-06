from torch.utils.data import SequentialSampler
import pytorch_lightning as pl

from alodataset import FlyingChairs2Dataset, Split
from alonet.raft.data_modules import Data2RAFT


class Chairs2RAFT(Data2RAFT):
    def __init__(self, args):
        self.val_names = ["Chairs2"]
        super().__init__(args)

    def train_dataloader(self):
        split = Split.VAL if self.train_on_val else Split.TRAIN
        dataset = FlyingChairs2Dataset(split=split, transform_fn=self.train_transform, sample=self.sample)
        sampler = SequentialSampler if self.sequential else None
        return dataset.train_loader(batch_size=self.batch_size, num_workers=self.num_workers, sampler=sampler)

    def val_dataloader(self):
        dataset = FlyingChairs2Dataset(split=Split.VAL, transform_fn=self.val_transform, sample=self.sample)

        return dataset.train_loader(batch_size=1, num_workers=self.num_workers, sampler=SequentialSampler)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser = Chairs2RAFT.add_argparse_args(parser)
    args = parser.parse_args(["--batch_size=8", "--num_workers=1"])
    multi = Chairs2RAFT(args)
    frames = next(iter(multi.train_dataloader()))
    frames[0].get_view().render()
    frames = next(iter(multi.val_dataloader()))
    frames[0].get_view().render()
