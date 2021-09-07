Coco detection
--------------
This module allows to read multiple datasets in **COCO JSON** format from a relative path stored in the 
**~/.aloception/alodataset_config.json** file. By default, if the configuration file does not exist or the database directory
is not found, the module will perform a command line user prompt to store/overwrite *alodataset_config.json* file
and the new database directory.

.. seealso:: 
   * `CocoDetection <https://pytorch.org/vision/0.8/datasets.html#detection>`_ pytorch module
   * :doc:`/alodataset/base_dataset` module
   * :doc:`/tutorials/data_setup` tutorial.

Basic use
=========
In order to use the module, :attr:`img_folder` and :attr:`ann_file` must be given as a minimum requirements::

   from alodataset import CocoDetectionDataset
   coco_dataset = CocoDetectionDataset(
      img_folder = "val2017", 
      ann_file = "annotations/instances_val2017.json"
   )

If the database does not exist or is wrong, a command line user prompt was executed: 

.. code-block:: bash
   :linenos:

   [WARNING] "coco" does not exist in config file. Please write the coco root directory: "user prompt"
   [WARNING] "old_path_directory" path does not exists for dataset: "coco". Please write the new directory: "user prompt" 

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

.. note:: 
   The dataset and paths of `COCO detection 2017 dataset <https://cocodataset.org/#detection-2017>`_ were taken as example.

Change of default database
==========================
:attr:`name` attribute is equal to **coco** as default. Another dataset can be configurated
using the relevant paths and a decided name. An example of how use 
`Cottontail-Rabbits Dataset <https://public.roboflow.com/object-detection/cottontail-rabbits-video-dataset>`_ is showing below::

   from alodataset import CocoDetectionDataset
   coco_dataset = CocoDetectionDataset(
      img_folder = "valid", 
      ann_file = "valid/_annotations.coco.json"
   )

.. seealso::
   For a custom dataset, see :doc:`How setup your data </tutorials/data_setup>` tutorial.

Class filtering
===============
:doc:`/alodataset/coco_detection_dataset` allows to select a desired set of classes, automatically managing the amount, labeling and ID change 
corresponding to the case. The classes to be filtered are exposed by :attr:`classes` parameter which by default does not filter 
the dataset to any specific class. 

For finetune applications, it is usual to manipulate databases in order to get a single class (for example, *people*)::

   from alodataset import CocoDetectionDataset
   coco_dataset = CocoDetectionDataset(
      classes = ["person"], 
      img_folder = "val2017", 
      ann_file = "annotations/instances_val2017.json"
   )

Now, :attr:`coco_dataset` should handled images with boxes that have *person* as label class.

.. warning:: 
   Each element in the desired list of :attr:`classes` **must be one element in** :attr:`CATEGORIES` **attribute**.   

Download a minimal dataset sample
=================================
For quick configuration issues, :mod:`coco_detection_dataset` has the option to download and use a sample of 8 examples 
of the original dataset, without the need to download the entire dataset. To download and/or access these samples, 
:attr:`sample` parameter has been declared::

   from alodataset import CocoDetectionDataset
   coco_dataset = CocoDetectionDataset(sample = True)

   for frame in coco_dataset.stream_loader():
      # Render each image
      frame.get_view().render()

.. warning::
   The purpose of the sample is to verify the correct functioning of the module. 
   It is not advisable to use such small samples for training.

Coco detection API
==================

.. automodule:: alodataset.coco_detection_dataset
   :members: CocoDetectionDataset
   :undoc-members:
   :show-inheritance:
