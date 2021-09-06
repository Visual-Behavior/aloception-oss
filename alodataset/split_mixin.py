from alodataset import Split


class SplitMixin(object):

    SPLIT_FOLDERS = {}

    def __init__(self, split: Split = Split.TRAIN, **kwargs):
        """Sequence Mixin
        Parameters
        ----------
        split: alodataset.Split
            Split.TRAIN by default.
        """
        super(SplitMixin, self).__init__(**kwargs)

        assert isinstance(split, Split)

        self.split = split

    def get_split_folder(self):
        assert self.split in self.SPLIT_FOLDERS
        return self.SPLIT_FOLDERS[self.split]
