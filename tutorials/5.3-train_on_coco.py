from argparse import ArgumentParser

import alonet
from alonet.detr import CocoDetection2Detr, DetrR50Finetune, LitDetr
from alonet.callbacks import MetricsCallback, ApMetricsCallback
from alonet.detr import DetrObjectDetectorCallback

import torch
from pytorch_lightning.callbacks import EarlyStopping

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

######
# Aloception Intro
######
#
# Aloception is developed under the pytorchlightning framework.
# For its use, aloception provides different modules that facilitate its use.
# For more information, see https://www.pytorchlightning.ai/


######
# Train process
######
#
# Based on pytorchlightning Trainer moduler
# Train DetrR50 over two classes (finetunning)
# We need to define a LightningDataModule and LightningModule
#
# Some ways to execute the tutorial:
# >> python tutorials/5.3-train_on_coco.py --save # Save the model in ~/.aloception/{project_run_id}/{run_id}
# >> python tutorials/5.3-train_on_coco.py --size 300 300 --batch_size 1 --no_augmentation # Reduce computational cost
# >> python tutorials/5.3-train_on_coco.py --cpu # Run train on cpu
#
# If you have multiples GPU, run (with n the decide GPU to use):
# >> CUDA_VISIBLE_DEVICES=n, python tutorials/5.3-train_on_coco.py
#
# All information available at:
#   [x] Trainer module: https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html
#   [x] Callbacks in pytorch lightning: https://pytorch-lightning.readthedocs.io/en/latest/extensions/callbacks.html
#   [x] train_on_coco parameters: Execute the command line
#       >> python alonet/detr/train_on_coco.py -h

# Parameters definition
# Build parser (concatenates arguments to modify the entire project)
parser = ArgumentParser(conflict_handler="resolve")
parser = CocoDetection2Detr.add_argparse_args(parser)
parser = LitDetr.add_argparse_args(parser)
parser = alonet.common.add_argparse_args(parser)  # Add common arguments in train process
args = parser.parse_args()
args.limit_train_batches = 1000  # Train only with 1000 samples. Comment if you want full-training

# Dataset use to train
# Init dataset loader and model with pre-trained weights loaded for fast train results
coco_loader = CocoDetection2Detr(
    args, classes=["cat", "dog"]
)  # Using COCODataset, background_class must have id = len(classes)
detr_ftune_nn = DetrR50Finetune(num_classes=2, weights="detr-r50")  # num_classes include bg_class
lit_detr = LitDetr(args, model=detr_ftune_nn)

# Callbacks
# Use by default if callbacks=None
obj_detection_callback = DetrObjectDetectorCallback(val_frames=next(iter(coco_loader.val_dataloader())))
metrics_callback = MetricsCallback()
ap_metrics_callback = ApMetricsCallback()
# User callbacks definition
early_callback = EarlyStopping(monitor="val_loss", patience=3)
callbacks = [obj_detection_callback, metrics_callback, ap_metrics_callback, early_callback]
# For more info about callbacks: https://pytorch-lightning.readthedocs.io/en/latest/extensions/callbacks.html

# Train process
args.max_epochs = 10  # Fast training (pytorchlightning Trainer parameter)
lit_detr.run_train(
    data_loader=coco_loader, args=args, project="detr_finetune", expe_name="coco_detr", callbacks=callbacks
)
lit_detr.model = lit_detr.model.eval().to(device)

# Check a random result
frame = next(iter(coco_loader.val_dataloader()))
frame = frame[0].batch_list(frame).to(device)
pred_boxes = lit_detr.inference(lit_detr(frame))[0]  # Inference from forward result
gt_boxes = frame[0].boxes2d

frame.get_view(
    [gt_boxes.get_view(frame[0], title="Ground truth boxes"), pred_boxes.get_view(frame[0], title="Predicted boxes"),]
).render()
