Detr: Data connectors
=====================

`Pytorch Lightning Data Module <https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html>`_
connector between dataset and model. Load train/val/test sets and make the preprocessing required for use by the
|detr|_.

Data2Detr
--------------------------------------

.. automodule:: alonet.detr.data_modules.data2detr
   :members:
   :undoc-members:
   :show-inheritance:

CocoDetection2Detr
--------------------------------------
LightningDataModule that make the connection between :mod:`CocoBaseDataset <alodataset.coco_detection_dataset>`
and :mod:`LitDetr <alonet.detr.train>` modules. See :mod:`Data2Detr <alonet.detr.data_modules.data2detr>`
to more information about the methods and configurations.

Examples
^^^^^^^^
.. code-block:: python

    from alonet.detr import CocoDetection2Detr
    from aloscene import Frame

    datamodule = CocoDetection2Detr(sample = True)

    train_frame = next(iter(datamodule.train_dataloader()))
    train_frame = Frame.batch_list(train_frame).get_view().render()

    val_frame = next(iter(datamodule.val_dataloader()))
    val_frame = Frame.batch_list(val_frame).get_view().render()


.. automodule:: alonet.detr.data_modules.coco_detection2detr
   :members:
   :undoc-members:
   :show-inheritance:

CocoPanoptic2Detr
--------------------------------------
LightningDataModule that make the connection between :mod:`CocoPanopticDataset <alodataset.coco_panoptic_dataset>`
and :mod:`LitPanopticDetr <alonet.detr_panoptic.train>` modules. See
:mod:`Data2Detr <alonet.detr.data_modules.data2detr>` to more information about the methods and configurations.

Examples
^^^^^^^^
.. code-block:: python

    from alonet.detr import CocoPanoptic2Detr
    from aloscene import Frame

    datamodule = CocoPanoptic2Detr(sample = True)

    train_frame = next(iter(datamodule.train_dataloader()))
    train_frame = Frame.batch_list(train_frame).get_view().render()

    val_frame = next(iter(datamodule.val_dataloader()))
    val_frame = Frame.batch_list(val_frame).get_view().render()


.. automodule:: alonet.detr.data_modules.coco_panoptic2detr
   :members:
   :undoc-members:
   :show-inheritance:

.. Hyperlinks
.. |detr| replace:: Detr-based architectures
.. _detr: detr_models.html#module-alonet.detr.detr
