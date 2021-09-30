Models
======

Basic usage
--------------------

Given that :mod:`~alonet.detr_panoptic.detr_panoptic` implements the Panoptic Head for a |detr|_, first the module
have to implement and be passed as :class:`~alonet.detr_panoptic.detr_panoptic.PanopticHead` parameter:

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

.. Hyperlinks
.. |detr| replace:: Detr-based models
.. _detr: detr_models.html#module-alonet.detr.detr
.. |detrfine| replace:: DetrFinetune models
.. _detrfine: detr_models.html#module-alonet.detr.detr_r50_finetune
