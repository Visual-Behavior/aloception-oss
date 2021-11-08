How to setup your data?
-----------------------
This tutorial explains how to setup a custom dataset to train a deep model using :doc:`Aloception framework </index>`.

.. admonition:: Goals
    :class: important

    1. Prepare your data using :doc:`Aloscene </aloscene/aloscene>` tools for 2D boxes prediction in classification tasks
    2. Use a high-level tool in :doc:`Alodataset </alodataset/alodataset>` to setup your data in **COCO JSON** format
    3. Manually set up paths for multiple datasets.

1. Prepare your data
====================
Depending on the application, there are different ways of organizing the information. It is common, in
computer vision applications, to have a set of images to train, validate and/or test a model, as well as a set of
annotations about the important information to use in each image.
Several types of annotations allow to develop applications for detection, segmentation, interpretation and
verification of image content.

Therefore, it is reasonable to think of a data structure presented below::

    dataset
    ├── train
    |   ├── img_train_0.jpg
    |   ├── img_train_1.jpg
    |   ├── ...
    |   └── img_train_l.jpg
    ├── valid
    |   ├── img_val_0.jpg
    |   ├── img_val_1.jpg
    |   ├── ...
    |   └── img_val_m.jpg
    ├── test (optional)
    |   ├── img_test_0.jpg
    |   ├── img_test_1.jpg
    |   ├── ...
    |   └── img_test_n.jpg
    └── annotations
        ├── ann_train
        |   ├── ann_img_train_0.txt
        |   ├── ann_img_train_1.txt
        |   ├── ...
        |   └── ann_img_train_l.txt
        └── ann_valid
            ├── ann_img_val_0.txt
            ├── ann_img_val_1.txt
            ├── ...
            └── ann_img_val_m.txt

.. hint::
    Recent database structures implement a single annotation file for each dataset. Examples are found in
    `COCO JSON <https://roboflow.com/formats/coco-json>`_ or
    `Tensorflow Object Detection CSV <https://roboflow.com/formats/tensorflow-object-detection-csv>`_
    formatted databases.

.. _labels:

Labels settings
"""""""""""""""
In many object detection applications is necessary to classify each prediction made into different categories.
For this, the :doc:`/aloscene/labels` module in :doc:`/aloscene/aloscene` provides one way to interpret the labels::

    import torch
    from aloscene import Labels

    example_labels = [0,1,2,0,1,0,1]
    example_label_names = ["person", "cat", "dog"]
    labels = Labels(
        torch.tensor(example_labels).to(torch.float32),
        labels_names = example_label_names,
        encoding = "id" # Also, we can use "one-hot"
    )

In this example, we take an encoding by ID of three objects:

   1. **person** label with **id = 0**
   2. **cat** label with **id = 1**
   3. **dog** label with **id = 2**

In additional, the first box should contain a *"person"*, the second one a *"cat"*, the third one a *"dog"*, and so on.
This corresponds to the order of the input tensor we assigned.

.. seealso::
    See :doc:`/aloscene/labels` for more information of its properties and attributes.

2D boxes declaration
""""""""""""""""""""
On the other hand, :doc:`/aloscene/aloscene` implements a module to handled the boxes. This module is called
:doc:`/aloscene/bounding_boxes_2d`::

    import torch
    from aloscene import BoundingBoxes2D

    random_boxes = torch.rand(labels.size(0), 4)

    # First option
    boxes = BoundingBoxes2D(
        random_boxes,
        boxes_format="xcyc",
        absolute=False,
        labels=labels
    )

    # Second option
    boxes = BoundingBoxes2D(
        random_boxes,
        boxes_format="xcyc",
        absolute=False,
    )
    boxes.append_labels(labels)

For the example, a random boxes set were implement with normalized values, (x_center, y_center, width, height)
as coordinates configuration and labels defined in :ref:`labels` section.

.. warning::
    If labels are decided, **labels.size(0) == boxes.size(0)**.

.. seealso::
    See :doc:`/aloscene/bounding_boxes_2d` for more information.

.. _setup-custom-dataset:

Frame implementation
""""""""""""""""""""
Given the possibility of that one frame can have multiple boxes, :doc:`/aloscene/frame` module has an attribute called
:attr:`boxes2d`. For a random image, we could use :doc:`/aloscene/frame` as follows for the previous boxes::

    import torch
    from aloscene import Frame

    image_size = (300,300) # A random image size
    frame = Frame(
        torch.rand(3, *image_size),
        names=("C", "H", "W"),
        boxes2d=boxes,
        normalization="01"
    )

.. hint::
    There are many ways to interpret the information in an image, but for purpose of this tutorial, we just implemented the boxes2d.
    See :doc:`/aloscene/frame` to read more about them.

BaseDataset module
""""""""""""""""""
:doc:`/alodataset/base_dataset` is a module based on `pytorch dataset class`_. It is able to handled a dataset **based on its root directory**.
For this, it saves the root directory in one configuration file named **aloception_config.json**, saved in **$HOME/.aloception** folder.

.. tip::
    If :doc:`/alodataset/base_dataset` module is used, *aloception_config.json* and **$HOME/.aloception** folder
    will be created automatically. However, :ref:`setup-multiples-datasets` section shows more details about that.

A general use of the module would be described by the following scheme of code:

.. code-block:: python

    import os

    from alodataset import BaseDataset
    from aloscene import Frame, BoundingBoxes2D, Labels

    class CustomDataset(BaseDataset):

        def __init__(self, name: str, image_folder: str, ann_folder: str, **kwargs):
            super().__init__(name, **kwargs)
            self.image_folder = os.path.join(self.dataset_dir, image_folder)
            self.ann_folder = os.path.join(self.dataset_dir, ann_folder)
            self.items = self._match_image_ann(self.img_folder, self.ann_folder)

        def _match_image_ann(self, img_folder, ann_folder):
            """TODO: Perform a function to match each image with theirs annotations.
            A minimal example could be the below: """
            return list(zip(os.listdir(image_folder), os.listdir(ann_folder)))

        def _load_image(self, id: int) -> Frame:
            """TODO: Load the image corresponds to 'id' input from 'self.image_folder'.
            Use self.items attribute! """
            pass

        def _load_ann(self, id: int) -> BoundingBoxes2D:
            """TODO: Load the annotations corresponds to 'id' input from 'self.ann_folder'.
            Use self.items attribute! """
            pass

        def getitem(self, idx: int) -> Frame:
            """TODO: Load the image corresponds to 'id' input from 'self.image_folder'"""
            frame = self._load_image(idx)
            boxes2d = self._load_ann(idx)
            frame.append_boxes2d(boxes2d)
            return frame

    data = CustomDataset(
        name="my_dataset",
        image_folder="path/image/folder",
        ann_folder="path/annotation/folder",
        transform_fn=lambda d: d, # Fake transform to give one example
    )

    for frame in data.stream_loader():
        frame.get_view().render()

.. important::
    There are many key concepts in :class:`BaseDataset` class:

    * We recommend to use :attr:`self.dataset_dir` attribute to get the dataset root folder. Also, we should define all the paths as relative from it.
    * All information required about each element in dataset will have to be given by :func:`getitem` function.
    * By default, the dataset size is given by :attr:`len(self.items)`.
    * Use :func:`stream_loader` and :func:`train_loader` to get individual samples or batch samples, respectively.

If an application must handled several datasets (like train, valid, test sets), we recommend using the :mod:`alodataset.SplitMixin` module:

.. code-block:: python

    from alodataset import Split, SplitMixin

    class CustomDatasetSplitMix(CustomDataset, SplitMixin):

        # Mandatory parameter with relative paths
        SPLIT_FOLDERS = {
            Split.VAL : "val2017",
            Split.TRAIN : "train2017",
            Split.TEST : "test2017",
        }

        def __init__(self, name: str, split: Split = Split.TRAIN, **kwargs):
            super(CustomDatasetSplitMix, self).__init__(name = name, **kwargs)
            self.image_folder = os.path.join(self.image_folder, self.get_split_folder())
            self.ann_folder = os.path.join(self.ann_folder, self.get_split_folder())

    data = CustomDataset(
        name="my_dataset",
        image_folder="path/image/folder",
        ann_folder="path/annotation/folder",
        split=Split.VAL
    )

.. note::
    CustomDatasetSplitMix could be developped in one class that used BaseDataset and SplitMixin classes.

.. hint::
    :class:`BaseDataset` class is based on `torch.utils.data.Dataset <pytorch dataset class>`_ module. All information
    is provided in :doc:`/alodataset/base_dataset`.

2. Setup a custom dataset based on COCO JSON
============================================
Many **COCO JSON** formatted datasets are available on the Internet. For example, roboflow_ provides several labeled and
configured datasets for implementation in machine learning applications. Also, there are many examples of how to setup
a dataset using **COCO JSON** format. `Create COCO Annotations From Scratch <https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch>`_
page explains how to make this work *manually*. Some tools is able to make and sort the annotation, like `roboflow annotations`_.

As a database handled tool on pytorch_ for object detection applications, :doc:`/index` offers a quick
configuration module on top of this type of database: :doc:`/alodataset/coco_detection_dataset`.

.. note::
    For this part of the tutorial, |coco|_ was used as a custom dataset. On the other hand, we assumed that
    the dataset was downloaded and stored in **$HOME/data/coco** directory. However, it is possible used a datataset based on
    **COCO JSON** format changing :attr:`img_folder` and :attr:`ann_file` paths.

For |coco|_, the *valid* dataset can be implemented by::

    from alodataset import CocoBaseDataset

    coco_dataset = CocoBaseDataset(
        name = "coco", # Parameter by default, change for others datasets
        img_folder = "val2017",
        ann_file = "annotations/instances_val2017.json",
        mode = "valid"
    )

    for frame in data.stream_loader():
        frame.get_view().render()

Now the module could read and process the images `COCO2017 detection valid set <coco>`_.

If a speed setup is required, we could use :attr:`sample` attribute, without having to download the data set::

    coco_dataset = CocoBaseDataset(sample = True)

.. warning::
    :attr:`sample` feature only applies to datasets managed by :doc:`/alodataset/alodataset`

.. _setup-multiples-datasets:

3. Make a config file
=====================
All modules based on :mod:`BaseDataset` will execute a user prompt after the execution of its declaration if the root directory
of the database does not exist in **aloception_config.json** file. However, it might be useful to think about configuring this file
for multiple datasets.

First, we need to create the **aloception_config.json** in **$HOME/.aloception** directory. This file must contain the following information:

.. code-block:: json

    {
        "dataset_name_1": "paht/of/dataset_1",
        "dataset_name_2": "paht/of/dataset_2",
        "...": "...",
        "dataset_name_n": "paht/of/dataset_n"
    }

An example could be:

.. code-block:: json

    {
        "coco": "$HOME/data/coco",
        "rabbits": "/data/rabbits",
        "pascal2012": "/data/pascal",
        "raccoon": "$HOME/data/raccon"
    }

With this pre-set up, the reading of the directory will be done automatically according to the :attr:`name` value.

.. admonition:: What is next ?
    :class: note

    Learn how to train a model using your custom data in :doc:`training_detr` and :doc:`training_deformable_detr`
    tutorials.

.. Hyperlinks
.. |coco| replace:: COCO 2017 detection dataset
.. _coco: https://cocodataset.org/#detection-2017
.. _roboflow: https://public.roboflow.com/
.. _roboflow annotations: https://roboflow.com/#annotate
.. _pytorch: https://pytorch.org/
.. _pytorch dataset class: https://pytorch.org/docs/stable/data.html#dataset-types
