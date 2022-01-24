Models
======

Panoptic Head is a Pytorch module that implements a network to connect with the output of a
:doc:`Detr-based model <detr_models>`. This new module is able to predict a segmentation features, represented by
a binary mask for each object predicted by :doc:`Detr model <detr_models>`.

.. figure:: ../images/panoptic_head.png
  :width: 100%
  :alt: Panoptic Head model
  :class: with-shadow

  Block diagram of panoptic head model, taken from
  `End-to-End Object Detection with Transformers <https://arxiv.org/pdf/2005.12872.pdf>`_ paper

.. seealso::

   * :doc:`Mask </aloscene/mask>` object to know the data representation of predictions.
   * `End-to-End Object Detection with Transformers <https://arxiv.org/pdf/2005.12872.pdf>`_ paper to understand how works
     the architecture.

Basic usage
--------------------

Given that :mod:`~alonet.detr_panoptic.detr_panoptic` implements the Panoptic Head for a |detr|_, `DETR <detr>` mode
must be defined as one of the first input parameter:

   .. code-block:: python

      from alonet.detr import DetrR50
      from alonet.detr_panoptic import PanopticHead

      detr_model = DetrR50()
      model = PanopticHead(DETR_module=detr_model)

If you want to finetune from the model pretrained on COCO dataset, a |detrfine|_ must be used:

   .. code-block:: python

      from alonet.detr import DetrR50Finetune
      from alonet.detr_panoptic import PanopticHead

      detr_model = DetrR50Finetune(num_classes=250)
      model = PanopticHead(DETR_module=detr_model)

As experimental work, it is possible to couple |Deformable|_ to panoptic head, by its previous definition:

   .. code-block:: python

      from alonet.deformable_detr import DeformableDetrR50Finetune
      from alonet.detr_panoptic import PanopticHead

      detr_model = DeformableDetrR50Finetune(num_classes=250)
      model = PanopticHead(DETR_module=detr_model)

Or simply use the predefined models with resnet 50 backbone for DETR/Deformable DETR.

   .. code-block:: python

      from alonet.detr_panoptic import DetrR50Panoptic
      from alonet.deformable_detr_panoptic import DeformableDetrR50Panoptic

      # Use DetrR50 + PanopticHead and load pretrained weights
      model DetrR50Panoptic(weights="detr-r50-panoptic")

      # Use DeformableDetrR50Refinement + PanopticHead with its pretrained weights
      model DeformableDetrR50Panoptic(
         weights="deformable-detr-r50-panoptic-refinement",
         activation_fn="softmax",
         with_box_refine=True
      )

.. warning::

   * As mentioned above, the work made on Deformable DETR is experimental, therefore the performance achieved differs
     from that obtained with DETR. See |detrPerf|_ and |deformablePerf|_
   * Unlike the :doc:`Deformable DETR <deformable>`, the weights provided by :doc:`Aloception </index>` were trained
     for the :mod:`DeformableDetrR50Refinement <alonet.deformable_detr.deformable_detr_r50_refinement>` architecture.

To run an inference:

   .. code-block:: python

      from aloscene import Frame
      device = model.device # supposed that `model` is already defined as above

      # read image and preprocess image with Resnet normalization
      frame = Frame(IMAGE_PATH).norm_resnet()
      # create a batch from a list of images
      frames = aloscene.Frame.batch_list([frame])
      frames = frames.to(device)

      # forward pass
      m_outputs = model(frames)
      # get boxes and MASK as aloscene.BoundingBoxes2D and aloscene.Mask from forward outputs
      pred_boxes, pred_masks = model.inference(m_outputs)

      # Display the predicted boxes
      frame.append_boxes2d(pred_boxes[0], "pred_boxes")
      frame.append_segmentation(pred_masks[0], "pred_masks")
      frame.get_view([frame.boxes2d, frame.segmentation]).render()

.. important::
   PanopticHead network is able to predict the segmentation masks, follow by each box predicted for the
   |detr|_. Is for this reason that inference function return a new output: :attr:`pred_masks`.

Panoptic head API
-----------------------------------------------

.. automodule:: alonet.detr_panoptic.detr_panoptic
   :members:
   :undoc-members:
   :show-inheritance:

DetrR50 Panoptic
-----------------------------------------------

.. automodule:: alonet.detr_panoptic.detr_r50_panoptic
   :members:
   :undoc-members:
   :show-inheritance:

DetrR50 Panoptic Finetune
-----------------------------------------------

.. automodule:: alonet.detr_panoptic.detr_r50_panoptic_finetune
   :members:
   :undoc-members:
   :show-inheritance:

Deformable DetrR50 Panoptic
-----------------------------------------------

.. automodule:: alonet.deformable_detr_panoptic.deformable_detr_r50_panoptic
   :members:
   :undoc-members:
   :show-inheritance:

Deformable DetrR50 Panoptic Finetune
-----------------------------------------------

.. automodule:: alonet.deformable_detr_panoptic.deformable_detr_r50_panoptic_finetune
   :members:
   :undoc-members:
   :show-inheritance:

.. Hyperlinks
.. |detr| replace:: Detr-based models
.. _detr: detr_models.html#module-alonet.detr.detr
.. |deformable| replace:: Deformable-based models
.. _deformable: deformable_models.html#module-alonet.deformable_detr.deformable_detr
.. |detrfine| replace:: DetrFinetune models
.. _detrfine: detr_models.html#module-alonet.detr.detr_r50_finetune

.. _detrPerf: https://github.com/Visual-Behavior/aloception/tree/master/alonet/detr_panoptic
.. |detrPerf| replace:: DETR results
.. _deformablePerf: https://github.com/Visual-Behavior/aloception/tree/master/alonet/deformable_panoptic
.. |deformablePerf| replace:: Deformable DETR results
