Models
===================

Basic usage
--------------------

To instantiate a DETR R50 (resnet50 backbone):

   .. code-block:: python

      from alonet.detr import DetrR50
      model = DetrR50()

If you want to finetune from the model pretrained on COCO dataset:

   .. code-block:: python

      from alonet.detr import DetrR50Finetune
      # NUM_CLASS is the number of classes in your finetune
      model = DetrR50Finetune(num_classes=NUM_CLASS, weights="detr-r50")

To run inference:

   .. code-block:: python

      from aloscene import Frame
      device = model.device # supposed that `model` is already defined as above

      # read image and preprocess image with Resnet normalization
      frame = aloscene.Frame(PATH_TO_IMAGE).norm_resnet()
      # create a batch from a list of images
      frames = aloscene.Frame.batch_list([frame])
      frames = frames.to(device)

      # forward pass
      m_outputs = model(frames)
      # get predicted boxes as aloscene.BoundingBoxes2D from forward outputs
      pred_boxes = model.inference(m_outputs)
      # Display the predicted boxes
      frame.append_boxes2d(pred_boxes[0], "pred_boxes")
      frame.get_view([frame.boxes2d]).render()

Detr Base
-----------------------

.. automodule:: alonet.detr.detr
   :members:
   :undoc-members:
   :show-inheritance:


Detr R50
----------------------------

.. automodule:: alonet.detr.detr_r50
   :members:
   :undoc-members:
   :show-inheritance:


Detr R50 Finetune
--------------------------------------

.. automodule:: alonet.detr.detr_r50_finetune
   :members:
   :undoc-members:
   :show-inheritance:
