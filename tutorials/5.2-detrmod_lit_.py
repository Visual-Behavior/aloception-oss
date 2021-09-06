from argparse import ArgumentParser
import torch

from alonet.detr import LitDetr, DetrR50Finetune, CocoDetection2Detr
from aloscene.renderer import View

# LightningDataModule used for the pupurso of this tutorial
coco_loader = CocoDetection2Detr(batch_size=1)
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

######
# Aloception Intro
######
#
# Aloception is developed under the pytorchlightning framework.
# For its use, aloception provides different modules that facilitate its use.
# For more information, see https://www.pytorchlightning.ai/


######
# LitDetr
######
#
# pl.LightningModule for train pytorch models based on End-to-End Object Detection with Transformers (DETR) architectures
#
# All information available at:
#   [x] End-to-End Object Detection with DeTr: https://arxiv.org/abs/2005.12872
#   [x] LightningModule: https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
#   [x] LitDetr parameters: Execute the command line
#       >> python alonet/detr/train.py -h

# Level 1
# Create LightningModule with default parameters (model="detr-r50", weights=None)
lit_detr = LitDetr()


def pred_on_coco(lit_detr):
    # Read random image in COCO dataset
    frame = next(iter(coco_loader.val_dataloader()))
    frame = frame[0].batch_list(frame).to(device)

    # Image evaluates in model
    lit_detr.model = lit_detr.model.eval().to(device)  # Launch model in device and eval
    pred_boxes = lit_detr(frame)  # Forward step
    pred_boxes = lit_detr.inference(pred_boxes)  # Inference from forward result

    pred_boxes[0].get_view(frame[0], size=(500, 700)).render()


pred_on_coco(lit_detr)

# Level 2
# Define LightningModule choosing the architecture and LitDetr default parameters
# model = DETR50, replacing output layer with num_classes outputs for finetunning
my_detr = DetrR50Finetune(num_classes=2, weights="detr-r50")
#Load weights from ~/.aloception/weights/detr-r50/detr-r50.pth
lit_detr = LitDetr(model=my_detr)
pred_on_coco(lit_detr)  # Weights loades, but random output layer!

# NOTE: For a modified model (e.g. DetrR50Finetune), a raise Exception will be launched if we try to
# load the pre-trained weights using 'weights' attribute. We must load them previously in custom model.
# lit_detr = LitDetr(model=my_detr, weights="detr-r50") # raise error

# Level 3
# Use namespace to define attribute values
args = ArgumentParser()
args = LitDetr.add_argparse_args(args).parse_args()
args.weights = "detr-r50"
lit_detr = LitDetr(args)  # MODIFY THEM IN CONSOLE LINE!
print(lit_detr.weights, lit_detr.gradient_clip_val, lit_detr.accumulate_grad_batches, lit_detr.model_name)
# detr-r50 0.1 4 detr-r50
pred_on_coco(lit_detr)  # Good predictions (model and pre-trained weights match!)

# Level 4
# Combine previous approaches. Both parameters will be replaced
lit_detr = LitDetr(args, gradient_clip_val=0, accumulate_grad_batches=2)
lit_detr.model = lit_detr.model.eval().to(device)  # Launch model in device and eval

for data in coco_loader.val_dataloader():
    frames = data[0].batch_list(data).to(device)
    pred_boxes = lit_detr(frames)  # Forward step
    pred_boxes = lit_detr.inference(pred_boxes)  # Inference
    pred_boxes[0].get_view(frames[0], size=(500, 700)).render(View.CV)

# Level 5
# Subclassing for aloception and more details on data module: https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
