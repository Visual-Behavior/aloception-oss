Merging datasets
----------------

Basic usage
===========

The class :attr:`MergeDataset` is used to merge multiple datasets::

   from alodataset import FlyingThings3DSubsetDataset, MergeDataset
   dataset1 = FlyingThings3DSubsetDataset(sequence_size=2, transform_fn=lambda f: f["left"])
   dataset2 = FlyingThings3DSubsetDataset(sequence_size=2, transform_fn=lambda f: f["right"])
   dataset = MergeDataset([dataset1, dataset2])

It is then possible to shuffle the datasets together, and sample batches than can contain items from different datasets::

   # this batch can contain items from dataset1 and/or dataset2
   batch = next(iter(dataset.train_loader(batch_size=4)))

It is possible to apply specific transformations to each dataset,
and then apply the same global transformation to the items::

   from alodataset.transforms import RandomCrop
   dataset1 = FlyingThings3DSubsetDataset(sequence_size=2, transform_fn=lambda f: f["left"]) # specific transform
   dataset2 = FlyingThings3DSubsetDataset(sequence_size=2, transform_fn=lambda f: f["right"]) # specific transform
   dataset = MergeDataset([dataset1, dataset2], transform_fn=RandomCrop(size=(368, 496)) # global transform

MergeDataset API
================

.. automodule:: alodataset.merge_dataset
   :members:
   :undoc-members:
   :show-inheritance:
