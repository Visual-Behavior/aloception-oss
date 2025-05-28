from torch.utils.data import SequentialSampler, RandomSampler

# import pytorch_lightning as pl

from alonet.raft.data_modules import Data2RAFT
from alodataset import FlyingThings3DSubsetDataset, Split
from alodataset import MergeDataset


class Things2RAFT(Data2RAFT):
    def __init__(self, args):
        self.val_names = ["Things"]
        super().__init__(args)

    @staticmethod
    def adapt(frame, camera):
        return frame[camera]

    def train_dataloader(self):
        split = Split.VAL if self.train_on_val else Split.TRAIN
        datasets = []
        datasets.append(
            FlyingThings3DSubsetDataset(
                split=split,
                cameras=["left"],
                labels=["flow", "flow_occ"],
                sequence_size=2,
                sample=self.sample,
                transform_fn=lambda f: self.train_transform(f["left"]),
            )
        )
        datasets.append(
            FlyingThings3DSubsetDataset(
                split=split,
                cameras=["right"],
                labels=["flow", "flow_occ"],
                sequence_size=2,
                sample=self.sample,
                transform_fn=lambda f: self.train_transform(f["right"]),
            )
        )

        dataset = MergeDataset(datasets)

        sampler = SequentialSampler if self.sequential else RandomSampler
        return dataset.train_loader(batch_size=self.batch_size, num_workers=self.num_workers, sampler=sampler)

    def val_dataloader(self):
        dataset = FlyingThings3DSubsetDataset(
            split=Split.VAL,
            cameras=["left"],
            labels=["flow", "flow_occ"],
            sequence_size=2,
            sample=self.sample,
            transform_fn=lambda f: self.val_transform(f["left"]),
        )

        return dataset.train_loader(batch_size=1, num_workers=self.num_workers, sampler=SequentialSampler)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser = Things2RAFT.add_argparse_args(parser)
    args = parser.parse_args(["--batch_size=8", "--num_workers=1"])
    multi = Things2RAFT(args)
    # 1 sample from train
    frames = next(iter(multi.train_dataloader()))
    frames[0].get_view().render()
    # 1 sample from val
    frames = next(iter(multi.val_dataloader()))
    frames[0].get_view().render()
