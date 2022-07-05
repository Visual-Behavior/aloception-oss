import torch
import random
import numpy as np

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
    weights : List[int]
        How to order samples.
            Example: MergeDataset([ds1, ds2, ds3], weights=[1, 2, 1]) will order samples
                     as follow : sample_ds1, sample_ds2, sample_ds2, sample_ds3, sample_ds1 ...
    lim_samples : int
        Maximum number of samples. Only when weights is not None.
    transform_fn : function
        transformation applied to each sample
    weights: List[int] | None
        For N datasets, a list of N integer weights.
        The samples from a dataset with weight `w` will appear `w` times in the MergeDataset.
    """

    def __init__(
            self,
            datasets,
            weights=None,
            shuffle=False,
            lim_samples=None,
            transform_fn=None,
            ):
        self.weights = weights
        self.shuffle = shuffle
        self.datasets = datasets
        self.ds_lengths = list(map(lambda x: len(x), datasets))

        if weights is not None:
            assert len(weights) == len(datasets), "weights and datasets must have the same length"
            max_length = self._init_max_length()
            lim_samples = max_length if lim_samples is None else min(lim_samples, max_length)
            print(f"[INFO] merging {len(datasets)} datasets of lengths" +
                "{' & '.join(map(lambda x: str(len(x)), datasets))}. Total length set to {lim_samples}")

        self.lim_samples = lim_samples
        self.transform_fn = transform_fn
        self.indices = self._init_indices()

    def _init_max_length(self):
        occ_rates = [len(ds) / occ for ds, occ in zip(self.datasets, self.weights)]
        short_idx = np.argmin(occ_rates)
        repeat_ds = len(self.datasets[short_idx]) // self.weights[short_idx]
        return sum([repeat_ds * occ for occ in self.weights])

    def _init_weights(self, weights):
        n_datasets = len(self.datasets)
        if weights is None:
            return [1] * n_datasets

        if len(weights) != n_datasets:
            raise RuntimeError("The number of weights should be equal to the number of datasets.")

        if any(type(w) != int for w in weights):
            raise RuntimeError("weights should be a list of int.")
        return weights

    def _init_indices(self):
        indices = []
        if self.weights is None:
            for dset_idx, dset in enumerate(self.datasets):
                for idx in range(len(dset)):
                    indices.append((dset_idx, idx))
        else:
            sample = 0
            ds_ranges = [list(range(l)) for l in self.ds_lengths]
            ds_pointers = [0 for _ in range(len(self.datasets))]
            if self.shuffle:
                [random.shuffle(l) for l in ds_ranges]
            while(sample < self.lim_samples):
                for dset_idx, occ in enumerate(self.weights):
                    for _ in range(occ):
                        indices.append((dset_idx, ds_ranges[dset_idx][ds_pointers[dset_idx]]))
                        ds_pointers[dset_idx] += 1
                        sample += 1


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
