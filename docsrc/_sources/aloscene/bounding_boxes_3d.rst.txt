BoundingBoxes3D
======================

Bounding Boxes 3D Tensor of shape (n, 7) of which the last dimension is : [xc, yc, zc, Dx, Dy, Dz, heading]

- Coordinate xc, yc, zc of boxes’ center
- Boxes’ dimension Dx, Dy, Dz along the 3 axis
- Heading is the orientation by rotating around vertical Y-axis.

With this coordinate system convention:

- The X axis is positive to the right
- The Y axis is positive downwards
- The Z axis is positive forwards

.. toctree::
   :maxdepth: 0
   :caption: Basic usage:

   notebooks/bounding_boxes_3d.ipynb


API
--------

.. automodule:: aloscene.bounding_boxes_3d
   :members:
   :undoc-members:
   :show-inheritance:
