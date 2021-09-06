from argparse import ArgumentParser
from alonet.detr import CocoDetection2Detr

from aloscene.renderer import View

# Procedure based on previous tutorials to display images
def img_plot(frames):
    frames = frames[0].batch_list(frames)
    frames.get_view(
        frames.boxes2d,
    ).render()


######
# Aloception Intro
######
#
# Aloception is developed under the pytorchlightning framework.
# For its use, aloception provides different modules that facilitate its use.
# For more information, see https://www.pytorchlightning.ai/


######
# CocoDetection2Detr
######
#
# pl.LightningDataModule for reading and preprocessing the images in coco dataset.
# Use CocoDetectionDataset (provided as an alodataset) to load images.
# Also, CocoDetection2Detr allows to load, clean, apply transformers and group images in batches for train/eval.
#
# All information available at:
#   [x] COCO dataset: https://cocodataset.org
#   [x] LightningDataModule: https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
#   [x] CocoDetection2Detr parameters: Execute the command line
#       >> python alonet/detr/coco_data_modules.py -h


# Level 1
# Create LightningDataModule to COCO dataset with default parameters
coco_loader = CocoDetection2Detr()
frames = next(iter(coco_loader.val_dataloader()))
img_plot(frames)

# Level 2
# Configure some class parameters
coco_loader = CocoDetection2Detr(batch_size=4)
frames = next(iter(coco_loader.val_dataloader()))
img_plot(frames)

# Level 3
# Use namespace to define attribute values
args = ArgumentParser()
args = CocoDetection2Detr.add_argparse_args(args).parse_args()  # Get parameters by default
# Example of change in value
args.no_augmentation = True
args.size = (800, 480)
print("args=", args)
# args= Namespace(batch_size=2, no_augmentation=True, num_workers=8, size=(800, 480), train_on_val=False)
coco_loader = CocoDetection2Detr(args)
frames = next(iter(coco_loader.val_dataloader()))
img_plot(frames)

# Level 4
# Combine previous approaches. 'batch_size' and 'classes' overwritten
coco_loader = CocoDetection2Detr(args, batch_size=4, classes=["dog", "cat"])

# A COMMON FULL-READ EXAMPLE
for frames in coco_loader.val_dataloader():
    frames = frames[0].batch_list(frames)
    frames.get_view(
        frames.boxes2d,
    ).render(View.CV)

# NOTE: 'classes' elements must be in COCO_CLASSES list
print("[INFO] Possible classes list:", coco_loader.CATEGORIES)
# [INFO] Possible classes list: {'bear', 'sports ball', 'handbag', 'car', 'chair', ... , 'keyboard', 'refrigerator', 'banana', 'apple'}

# Level 5
# Subclassing for aloception and more details on data module: https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
