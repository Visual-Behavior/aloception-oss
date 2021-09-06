OrientedBoxes2D
====================

Oriented Boxes 2D is defined by [x, y, w, h, theta] in which:

    - x, y: center coordinates (could be relative or absolute values)
    - w, h: width, height (could be relative or absolute values)
    - theta: rotation angle

Basic usage
---------------

.. code-block:: python

    import numpy as np
    import torch
    from aloscene import OrientedBoxes2D

    boxes = OrientedBoxes2D(
        torch.tensor(
            [
                # [x,   y,   w,   h, theta]
                [0.5, 0.5, 0.2, 0.2, 0],
                [0.1, 0.1, 0.2, 0.3, np.pi / 6],
                [0.1, 0.8, 0.1, 0.3, -np.pi / 3],
                [0.6, 0.3, 0.4, 0.2, np.pi / 4],
            ],
            device=torch.device("cuda"),
        ),
        absolute=False, # use relative values for x, y, w, h
    )

    boxes.get_view().render()

Get coordinates of 4 corners for each boxes:

.. code-block:: python

    print(boxes.corners())

Convert to absolute value format with frame size = (300, 300):

.. code-block:: python

    abs_boxes = boxes.abs_boxes((300, 300))
    print(abs_boxes)
    print(abs_boxes.absolute)  # True
    print(abs_boxes.rel_pos() == boxes) # True

Calucate oriented IoU/GIoU with another set of boxes:

.. code-block:: python

    boxes2 = OrientedBoxes2D(
        torch.tensor(
                [
                    [1, 1, 2, 2, 0],
                    [5, 5, 2, 3, np.pi / 6],
                    [1, 1, 1, 3, -np.pi / 3],
                    [3, 1, 4, 2, np.pi / 4]
                ],
                device=torch.device("cuda")
        )
    )
    iou = boxes.rotated_iou_with(boxes2)
    giou = boxes.rotated_giou_with(boxes2)
    print(iou)
    print(giou)


API
---------------------

.. automodule:: aloscene.oriented_boxes_2d
   :members:
   :undoc-members:
   :show-inheritance:
