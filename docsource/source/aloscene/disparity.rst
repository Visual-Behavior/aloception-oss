Disparity
#########

The aloscene :attr:`Disparity` object represents a Disparity Map.

Basic Use
*********

Creating a Disparity object
===========================

A :attr:`Disparity` object can be initialized from a path to a disparity file::

   from aloscene import Disparity
   disp = Disparity("path/to/disparity.pfm")

or from an existing tensor::

   import torch
   disp_tensor = torch.zeros((1,400,600))
   disp = Disparity(disp_tensor, names=("C","H","W"))

An occlusion mask can be attached to the :attr:`Disparity` during its creation::

   from aloscene import Mask
   occlusion = Mask(torch.zeros(1,400, 600), names=("C","H","W"))
   disp = Disparity("path/to/disparity.pfm", occlusion)

or later::

   disp.append_occlusion(occlusion)

Sign convention
===============

.. warning::
   The disparity between left and right image can be expressed in two different conventions:

   * "unsigned": absolute difference of pixels between left and right image
   * "signed": relative difference of pixels (negative for left image)

If :attr:`camera_side` is known, it is possible to switch from one convention to the other::

   disp = Disparity(torch.ones((1, 5, 5)), disp_format="unsigned", names=("C", "H", "W"), camera_side="left")
   disp_signed = disp.signed()
   print("unsigned:", disp.as_tensor().unique())
   print("signed": disp_signed.as_tensor().unique())

In this example, we switch from "unsigned" to "signed" for left camera, therefore the disparity become negative::

   unsigned: tensor([1.])
   signed: tensor([-1.])

Easy visualization
==================

With aloscene API, it is really straigthforward to visualize data. In one line of code,
you can get a RGB view of a :attr:`Disparity` object and its occlusion map and display it::

   from alodataset import SintelDisparityDataset
   # read disparity data
   dataset = SintelDisparityDataset()
   frame = next(iter(dataset.stream_loader()))
   disparity = frame["left"].disparity
   # visualize it
   view = disparity.get_view() # view containing a numpy array with RGB data
   view.render() # render with matplotlib or cv2 and show data in a new window

.. figure:: ../images/disp_view.jpg
   :scale: 50%

   Visualization of a disparity sample from MPI Sintel Disparity dataset.


Easy transformations
====================

With aloscene API, it is easy to apply transformations to visual data, even when it necessitate extra care.

For example, horizontally flipping a frame with disparity data is usually tedious.
The image, the disparity and the occlusion should be flipped. But the disparity necessitate a specific transformation:
the sign of disparity values must be inverted only if the disparity is expressed in relative units.

With aloscene API, this is done automatically in one function call::

   frame_flipped = frame["left"].hflip() # flip frame and every attached labels
   disp_flipped = frame_flipped.disparity
   disp_flipped.get_view().render() # display flipped flow and occlusion

.. figure:: ../images/disp_hflip.jpg
   :scale: 50%

   Visualization of the previous disparity sample, after horizontal flip. The colors are different,
   because disparity values have been correctly modified. The occlusion has also been flipped.

.. warning::
   When horizontally flipping frames with disparity labels, the "left" and "right" frames should be swapped.
   This is necessary to keep the disparity values consistent with the images content.


Disparity API
*************

.. automodule:: aloscene.disparity
   :members:
   :undoc-members:
   :show-inheritance:
