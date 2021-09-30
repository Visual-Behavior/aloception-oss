Mask
----

The aloscene :attr:`Mask` object represents a binary or float mask. It can be used to represent different objects,
for example occlusions mask for flow and disparity use cases, or binary masks for segmentation tasks

.. note::

   The values of a :attr:`Mask` are between 0. and 1., to allow to partially/completely mask another tensor
   by multiplying it by the mask.

Basic Use
=========

A :attr:`Mask` object can be initialized from a path to a mask file::

   from aloscene import Mask
   mask = Mask("path/to/mask.png")

or from an existing tensor::

   import torch
   mask_float_tensor = torch.rand((1,400,600))
   mask = Mask(mask_tensor, names=("C","H","W"))


Mask API
--------

.. automodule:: aloscene.mask
   :members:
   :undoc-members:
   :show-inheritance:
