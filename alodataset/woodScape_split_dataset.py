from alodataset import Split, SplitMixin
from alovb.datasets import WooodScapeDataset


class WoodScapeSplitDataset(WooodScapeDataset, SplitMixin):
    SPLIT_FOLDERS = {Split.VAL: -0.1, Split.TRAIN: 0.9}

    def __init__(
            self,
            split=Split.TRAIN,
            **kwargs
            ):
        self.split = split
        super().__init__(fragment=self.get_split_folder(), **kwargs)


if __name__ == "__main__":
    val = WoodScapeSplitDataset(split=Split.VAL)
    train = WoodScapeSplitDataset(split=Split.TRAIN)

    print("val :",  len(val))
    print("train :",  len(train))
