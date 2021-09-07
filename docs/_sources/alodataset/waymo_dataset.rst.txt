Waymo Dataset
=================

This module reads `Waymo Open Dataset <https://waymo.com/open/>`_ images, camera calibrations and labels 2D/3D.

Dataset preparation
-------------------------

Waymo Open Dataset is stored originally in compressed tar file containing TFRecord files.
Once the dataset is downloaded, it is needed to be decompressed and converted from TFRecord format to jpeg files and pickle files. 
This process is called dataset preparation and handled by the method `self.prepare()`.

For this preparation we need to install additional packages in `alodataset/prepare/waymo-requirements.txt`.

Steps to prepare Waymo Open Dataset:

   #. Download Waymo Open Dataset, decompressed it in a directory called `waymo`. The full path can be `YOUR_PATH/waymo`.
      This directory should have structure as follow:
      
      ::

         YOUR_PATH/waymo/
         |__testing
         |  |__ *.tfrecord
         |__training
         |  |__ *.tfrecord
         |__validation
            |__ *.tfrecord

   #. Add in `YOUR_HOME/.aloception/alodataset_config.json` a pair key-value

      .. code-block:: json

         "waymo" : "YOUR_PATH/waymo"

   #. From the aloception root, run:

      .. code-block:: none

         python alodataset/prepare/waymo_prepare.py

      This script will convert TFRecord files in a new directory `YOUR_PATH/waymo_prepared` and replace this new path in `YOUR_HOME/.aloception/alodataset_config.json`.
      This conversion can takes hours to complete depending on the system hardware. 
      In case it is stopped/killed in the middle of preparation, we can always resume the preparation by executing the script.

The new prepared directory will be as follow:

      :: 

         YOUR_PATH/waymo_prepared/
         |__testing
         |__training
         |__validation

      In each subdirectory, we have:

      ::

         |__calib.pkl
         |__camera_label.pkl
         |__lidar_label.pkl
         |__pose.pkl
         |__image0
         |__|__*.jpeg
         |__image1
         |__|__*.jpeg
         |__image2
         |__|__*.jpeg
         |__image3
         |__|__*.jpeg
         |__image4
         |__|__*.jpeg
         |__velodyne

- `calib.pkl` is a pickle file containing camera calibrations.
- `camera_label.pkl` is a pickle file containing labels 2D: boxes, classes, track id, camera id.
- `lidar_label.pkl` is a pickle file containing labels 3D: boxes 3d, boxes 3d projected on image, class, track id, camera id, speed, acceleration.
- `pose.pkl` is a pickle file containing vehicle pose.
- `imageX` folders contain images from camera `X` in jpeg format.
- `velodyne` directory should be empty.


Basic usage
----------------
.. code-block:: python

   from alodataset import Split, WaymoDataset

   waymo_dataset = WaymoDataset(
      # Split.VAL for data in `validation` directory, 
      # use Split.TRAIN/Split.TEST for `training`/`testing` directory
      split=Split.VAL, 
      labels=["gt_boxes_2d", "gt_boxes_3d"],
      sequence_size=3)

   # prepare waymo dataset from tfrecord file
   # if the dataset is already prepared, it simply checks the prepared dataset
   # this line is optional if the dataset is fully prepared
   waymo_dataset.prepare()

   for frames in waymo_dataset.train_loader(batch_size=2):
      # frames is a list (with len=batch size) of dict
      # dict key is camera name
      # dict value is a Frame of shape (t, c, h, w) with t sequence_size
      
      # convert a list of front camera's frames into a batch
      front_frames = Frame.batch_list([frame["front"] for frame in frames])
      print(front_frames.shape) # (b, t, c, h, w), in this case b=batch=2, t=sequence_size=3
      # access to labels
      print(front_frames.boxes2d)
      print(front_frames.boxes3d)
      print(front_frames.cam_intrinsic)
      print(front_frames.cam_extrinsic)


API
--------

.. automodule:: alodataset.waymo_dataset
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__