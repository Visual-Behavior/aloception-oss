.. _coco_panoptic:

Coco panoptic
--------------
This module allows to read multiple datasets in **COCO JSON** format, with
`panoptic annotations <https://kharshit.github.io/blog/2019/10/18/introduction-to-panoptic-segmentation-tutorial>`_
from a relative path stored in **~/.aloception/alodataset_config.json** file.

.. seealso::
   * :doc:`/alodataset/base_dataset` module
   * :doc:`/tutorials/data_setup` tutorial.

Basic use
=========
By default, :mod:`CocoPanopticDataset <alodataset.coco_panoptic_dataset>` follows the file structure shown below::

   dataset
      ├── train2017
      |   ├── img_train_0.jpg
      |   ├── img_train_1.jpg
      |   ├── ...
      |   └── img_train_L.jpg
      ├── valid2017
      |   ├── img_val_0.jpg
      |   ├── img_val_1.jpg
      |   ├── ...
      |   └── img_val_M.jpg
      └── annotations
         ├── panoptic_train2017.json
         ├── panoptic_val2017.json
         ├── panoptic_train2017
         |   ├── img_ann_train_0.jpg
         |   ├── img_ann_train_1.jpg
         |   ├── ...
         |   └── img_ann_train_L.jpg
         └── panoptic_val2017
            ├── img_ann_val_0.jpg
            ├── img_ann_val_1.jpg
            ├── ...
            └── img_ann_val_M.jpg

If the dataset handled the same files structure, the common use of
:mod:`CocoPanopticDataset <alodataset.coco_panoptic_dataset>` will be::

   from alodataset import CocoPanopticDataset, Split
   coco_dataset = CocoPanopticDataset(split = Split.VAL)

All paths for train/val/set are defined by :mod:`CocoPanopticDataset <alodataset.coco_panoptic_dataset>`. However, a
child class can be created from :mod:`CocoPanopticDataset <alodataset.coco_panoptic_dataset>` to change the default
paths or change the attributes before to call the class. There are two ways to change the images folder path:

.. code-block:: python

   from alodataset import CocoPanopticDataset, Split

   # First way
   class MyPanopticDataset(CocoPanopticDataset):
      SPLIT_FOLDERS = {
         Split.TRAIN: "new/train/image/folder/path",
         Split.VAL: "new/val/image/folder/path",
      }

   my_dataset = MyPanopticDataset(split = Split.VAL)

   # Second way
   CocoPanopticDataset.SPLIT_FOLDERS = {
      Split.TRAIN: "new/train/image/folder/path",
      Split.VAL: "new/val/image/folder/path",
   }
   my_dataset = CocoPanopticDataset(split = Split.TRAIN)

.. important::

   There are three main paths to take into account for each set:

   1. Images folder path (saves in :attr:`SPLIT_FOLDERS` attribute)
   2. Panoptic annorations file path (saves in :attr:`SPLIT_ANN_FILES` attribute)
   3. Panoptic images folder path (saves in :attr:`SPLIT_ANN_FOLDERS` attribute), with the respective id class
      annotation per pixel

After its correct initialization, the module allows the reading of individual images in two ways::

   # Get a frame by index
   frame0 = coco_dataset.getitem(0)
   # Get a random frame in batch
   framer = next(iter(coco_dataset.stream_loader()))

Also, a iterable object can be generated using :func:`train_loader` or :func:`stream_loader` functions::

   # Frame iteration by individual images
   for frame in coco_dataset.stream_loader():
      # Render each image
      frame.get_view().render()

   # Frames iteration by images in a decided batches
   for frame in coco_dataset.train_loader(batch_size = 2):
      # Transform each list in a batch, and then render it
      frames = Frame.batch_list(frames)
      frames.get_view().render()

.. seealso::

   Excluding the paths attributes, :ref:`coco_panoptic` follows the same input parameters than
   :doc:`coco_detection_dataset`.

Coco panoptic API
==================

.. automodule:: alodataset.coco_panoptic_dataset
   :members: CocoPanopticDataset
   :undoc-members:
   :show-inheritance:
