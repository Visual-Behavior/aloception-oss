Training
=========================

For training, :mod:`LitPanopticDetr <alonet.detr_panoptic.train>` implements a
`Pytorch Lightning Module <https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html>`_ that
uses as default the :mod:`~alonet.detr.detr_r50` module coupled with :mod:`~alonet.detr_panoptic.detr_panoptic`.
For this reason, :mod:`alonet.detr.criterion` and :mod:`alonet.detr.matcher` are used in the training. However,
the :mod:`alonet.detr_panoptic.callbacks` are adapted to the predictions of the masks in the inference process.

Training
------------------------------------

.. automodule:: alonet.detr_panoptic.train
   :members:
   :undoc-members:
   :show-inheritance:

Callbacks
----------------------------------------

.. automodule:: alonet.detr_panoptic.callbacks
   :members:
   :undoc-members:
   :show-inheritance:

.. Criterion
.. ----------------------------------------

.. .. automodule:: alonet.detr.criterion
..    :members:
..    :undoc-members:
..    :show-inheritance:


.. Matcher
.. --------------------------------------

.. .. automodule:: alonet.detr.matcher
..    :members:
..    :undoc-members:
..    :show-inheritance:
