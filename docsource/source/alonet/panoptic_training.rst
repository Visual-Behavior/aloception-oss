Training
=========================

For training, :mod:`LitPanopticDetr <alonet.detr_panoptic.train>` and
:mod:`LitPanopticDeformableDetr <alonet.deformable_detr_panoptic.train>` are
`Pytorch Lightning Modules <https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html>`_, with
:mod:`~alonet.detr_panoptic.detr_r50_panoptic` and :mod:`~alonet.deformable_detr_panoptic.deformable_detr_r50_panoptic`
as default architectures, respectively.

The :mod:`DETR Panoptic Criterion <alonet.detr_panoptic.criterion>` and
:mod:`Deformable DETR Panoptic Criterion <alonet.deformable_detr_panoptic.criterion>` inherit the
:mod:`DETR criterion <alonet.detr.criterion>`/:mod:`Deformable DETR criterion <alonet.deformable_detr.criterion>`,
adding to each one the :class:`~alonet.detr_panoptic.criterion.PanopticCriterion`.

Finally, :mod:`alonet.detr_panoptic.callbacks` adapts the segmentation predictions at the inference process on
training-loop, for both architectures.

.. note::

   :mod:`alonet.detr.matcher` and :mod:`alonet.deformable_detr.matcher` are the matcher used in training

LitPanopticDetr
------------------------------------

.. automodule:: alonet.detr_panoptic.train
   :members:
   :undoc-members:
   :show-inheritance:

LitPanopticDeformableDetr
------------------------------------

.. automodule:: alonet.deformable_detr_panoptic.train
   :members:
   :undoc-members:
   :show-inheritance:

Panoptic Criterion
----------------------------------------

.. autoclass:: alonet.detr_panoptic.criterion.PanopticCriterion
   :members:
   :undoc-members:
   :show-inheritance:

Detr Panoptic Criterion
----------------------------------------

This class computes the loss for :mod:`DETR_PANOPTIC <alonet.detr_panoptic.detr_panoptic>`. The process happens
in two steps:

1) We compute hungarian assignment between ground truth boxes and the outputs of the model
2) We supervise each pair of matched ground-truth / prediction (supervise class, boxes and masks).

.. autoclass:: alonet.detr_panoptic.criterion.DetrPanopticCriterion
   :members:
   :undoc-members:
   :show-inheritance:

Deformable Detr Panoptic Criterion
----------------------------------------

This class computes the loss for
:mod:`DEFORMABLE DETR PANOPTIC <alonet.deformable_detr_panoptic.deformable_detr_panoptic>`.
The process happens in two steps:

1) We compute hungarian assignment between ground truth boxes and the outputs of the model
2) We supervise each pair of matched ground-truth / prediction (supervise class, boxes and masks).

.. autoclass:: alonet.deformable_detr_panoptic.criterion.DeformablePanopticCriterion
   :members:
   :undoc-members:
   :show-inheritance:

Detr Panoptic Callbacks
----------------------------------------

.. automodule:: alonet.detr_panoptic.callbacks
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: alonet.deformable_detr_panoptic.callbacks
   :members:
   :undoc-members:
   :show-inheritance:

.. Matcher
.. --------------------------------------

.. .. automodule:: alonet.detr.matcher
..    :members:
..    :undoc-members:
..    :show-inheritance:

.. .. automodule:: alonet.deformable_detr.matcher
..    :members:
..    :undoc-members:
..    :show-inheritance:
