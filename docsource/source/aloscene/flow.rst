Flow
####

The aloscene :attr:`Flow` object represents a 2D Optical Flow Map.

Basic Use
*********

Creating a Flow object
======================

A :attr:`Flow` object can be initialized from a path to a flow file::

   from aloscene import Flow
   flow = Flow("path/to/flow.flo")

or from an existing tensor::

   import torch
   flow_tensor = torch.zeros((2,400,600))
   flow = Flow(flow_tensor, names=("C","H","W"))

An occlusion mask can be attached to the :attr:`Flow` during its creation::

   from aloscene import Mask
   occlusion = Mask(torch.zeros(1,400, 600), names=("C","H","W"))
   flow = Flow("path/to.flow.flo", occlusion)

Or later::

   flow.append_occlusion(occlusion)


Easy visualization
==================

With aloscene API, it is really straigthforward to visualize data. In one line of code,
you can get a RGB view of a :attr:`Flow` object and its occlusion map and display it::

   from alodataset import ChairsSDHomDataset
   # read flow data
   dataset = ChairsSDHomDataset(sample=True)
   frame = next(iter(dataset.stream_loader()))
   flow = frame[0].flow["flow_forward"]
   # visualize it
   view = flow.get_view() # view containing a numpy array with RGB data
   view.render() # render with matplotlib or cv2 and show data in a new window

.. figure:: ../images/flow_view.jpg
   :scale: 50%

   Visualization of an optical flow sample from ChairsSDHomDataset.


Easy transformations
====================

With aloscene API, it is easy to apply transformations to visual data, even when it necessitate extra care.

For example, horizontally flipping a frame with flow data is usually tedious.
The image, the flow and the occlusion should be flipped. But the flow necessitate a specific transformation:
the values of the vectors must be modified, in addition to changing their coordinates in the tensor.

With aloscene API, this is done automatically in one function call::

   frame_flipped = frame.hflip() # flip frame and every attached labels

   flow_flipped = frame_flipped[0].flow["flow_forward"]
   flow_flipped.get_view().render() # display flipped flow and occlusion

.. figure:: ../images/flow_hflip.jpg
   :scale: 50%

   Visualization of the previous flow sample, after horizontal flip. The colors are different,
   because flow vectors values have been correctly modified. The occlusion has also been flipped.


Flow API
********

.. automodule:: aloscene.flow
   :members:
   :undoc-members:
   :show-inheritance:
