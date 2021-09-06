from torch.utils.data import SequentialSampler

# import pytorch_lightning as pl

from alonet.raft.data_modules import Data2RAFT
from alodataset import SintelDataset, Split


class Sintel2RAFT(Data2RAFT):
    def __init__(self, args):
        self.val_names = ["Things"]
        super().__init__(args)

    def train_dataloader(self):
        if self.train_on_val:
            raise ValueError("No validation set for sintel dataset")

        dataset = SintelDataset(
            split=Split.TRAIN,
            cameras=["left"],
            labels=["flow", "flow_occ"],
            sequence_size=2,
            sample=self.sample,
            transform_fn=lambda f: self.train_transform(f["left"]),
        )

        sampler = SequentialSampler if self.sequential else None
        return dataset.train_loader(batch_size=self.batch_size, num_workers=self.num_workers, sampler=sampler)

    def val_dataloader(self):
        return None


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser = Sintel2RAFT.add_argparse_args(parser)
    args = parser.parse_args(["--batch_size=8", "--num_workers=1"])
    multi = Sintel2RAFT(args)
    # 1 sample from train
    frames = next(iter(multi.train_dataloader()))
    frames[0].get_view().render()
