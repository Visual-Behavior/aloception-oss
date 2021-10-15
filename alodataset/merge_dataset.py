import torch

from alodataset.base_dataset import rename_data_to_none
from alodataset.base_dataset import stream_loader, train_loader


class MergeDataset(torch.utils.data.Dataset):
    """
    Dataset merging multiple alodataset.BaseDataset

    Iterating sequentially over the dataset will yield
    all samples from first dataset, then all samples
    from the next dataset until the last dataset.

    Shuffling the dataset will shuffle the samples of all datasets together

    Parameters
    ----------
    datasets : List[alodataset.BaseDataset]
        List of datasets
    transform_fn : function
        transformation applied to each sample
    """

    def __init__(self, datasets, transform_fn=None):
        self.datasets = datasets
        self.indices = self._init_indices()
        self.transform_fn = transform_fn

    def _init_indices(self):
        indices = []
        for dset_idx, dset in enumerate(self.datasets):
            for idx in range(len(dset)):
                indices.append((dset_idx, idx))
        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        dset_idx, sample_idx = self.indices[idx]
        data = self.datasets[dset_idx][sample_idx]
        if self.transform_fn is not None:
            data = self.transform_fn(data)
            data = rename_data_to_none(data)
        return data

    def _collate_fn(self, batch_data):
        """data loader collate_fn"""
        return batch_data

    def stream_loader(self, num_workers=2):
        """Get a stream loader from the dataset. Compared to the :func:`train_loader`
        the :func:`stream_loader` do not have batch dimension and do not shuffle the dataset.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            Dataset to make dataloader
        num_workers : int
            Number of workers, by default 2

        Returns
        -------
        torch.utils.data.DataLoader
            A generator
        """
        return stream_loader(self, num_workers=num_workers)

    def train_loader(self, batch_size=1, num_workers=2, sampler=torch.utils.data.RandomSampler):
        """Get training loader from the dataset"""
        return train_loader(self, batch_size=batch_size, num_workers=num_workers, sampler=sampler)


if __name__ == "__main__":
    from alodataset import ChairsSDHomDataset, FlyingThings3DSubsetDataset, Split

    chairs = ChairsSDHomDataset(sample=True)
    flying = FlyingThings3DSubsetDataset(sample=True, transform_fn=lambda f: f["left"])

    multi = MergeDataset([chairs, flying])

    # after shuffling a mergedataset, a batch can contain samples from different datasets
    batch_size = 4
    for frame in multi.train_loader(batch_size=batch_size):
        for i in range(batch_size):
            frame[i].get_view().render()
        break
