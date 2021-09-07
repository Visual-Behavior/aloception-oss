Models
======

Basic usage
--------------------

To instantiate a Deformable DETR R50 (resnet50 backbone):

   .. code-block:: python

      from alonet.deformable_detr import DeformableDetrR50
      model = DeformableDetrR50(num_classes=NUM_CLASS)

To instantiate a Deformable DETR R50 (resnet50 backbone) with iterative box refinement:

   .. code-block:: python

      from alonet.deformable_detr import DeformableDetrR50Refinement
      model = DeformableDetrR50Refinement(num_classes=NUM_CLASS)

If you want to finetune from the model pretrained on COCO dataset:

   .. code-block:: python

      from alonet.deformable_detr import DeformableDetrR50Finetune
      # NUM_CLASS is the number of classes in your finetune
      model = DeformableDetrR50Finetune(num_classes=NUM_CLASS, weights="deformable-detr-r50")

   .. code-block:: python

      # with iterative box refinement
      from alonet.deformable_detr import DeformableDetrR50RefinementFinetune
      # NUM_CLASS is the number of classes in your finetune
      model = DeformableDetrR50RefinementFinetune(num_classes=NUM_CLASS, weights="deformable-detr-r50-refinement")

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

Deformable DETR Base
-----------------------------------------------

.. automodule:: alonet.deformable_detr.deformable_detr
   :members:
   :undoc-members:
   :show-inheritance:


Deformable DETR R50
----------------------------------------------------

.. automodule:: alonet.deformable_detr.deformable_detr_r50
   :members:
   :undoc-members:
   :show-inheritance:


Deformable DETR R50 with refinement
----------------------------------------------------------------

.. automodule:: alonet.deformable_detr.deformable_detr_r50_refinement
   :members:
   :undoc-members:
   :show-inheritance:

.. module:: alonet.deformable_detr.deformable_detr_r50_finetune

Deformable DETR R50 Finetune
----------------------------------------------------

.. autoclass:: alonet.deformable_detr.deformable_detr_r50_finetune.DeformableDetrR50Finetune
   :members:
   :undoc-members:
   :show-inheritance:


Deformable DETR R50 Finetune with refinement
----------------------------------------------------

.. autoclass:: alonet.deformable_detr.deformable_detr_r50_finetune.DeformableDetrR50RefinementFinetune
   :members:
   :undoc-members:
   :show-inheritance:
