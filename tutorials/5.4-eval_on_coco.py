from argparse import ArgumentParser

import torch

from alonet.common import load_training
from alonet.detr import CocoDetection2Detr, DetrR50Finetune, LitDetr

import aloscene
import alonet

######
# Aloception Intro
######
#
# Aloception is developed under the pytorchlightning framework.
# For its use, aloception provides different modules that facilitate its use.
# For more information, see https://www.pytorchlightning.ai/


######
# Eval process
######
#
# Load and evaluate the performarce of LitDetr trained module.
#
# All information available at:
#   [x] eval_on_coco parameters: Execute the command line
#       >> python alonet/detr/eval_on_coco.py -h

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# Build parser
parser = ArgumentParser(description="Evaluate performance DetrR50 model")
parser.add_argument("--ap_limit", type=int, default=None, help="Limit AP computation at the given number of sample")
parser.add_argument("--run_id", type=str, default=None, help="Load the weights from this saved experiment")
args = parser.parse_args()

# Choose dataset and create a empty model in order to load the weights
if args.run_id is None:  # Model by default
    coco_loader = CocoDetection2Detr(batch_size=1)
    lit_detr = LitDetr(weights="detr-r50")
else:  # Load model trained and saved with tutorials/5.3-train_on_coco.py program
    coco_loader = CocoDetection2Detr(batch_size=1, classes=["cat", "dog"])
    nn_skeleton = DetrR50Finetune(num_classes=2)
    lit_detr = load_training(LitDetr, args=args, model=nn_skeleton, run_id=args.run_id, project_run_id="detr_finetune")
lit_detr.model = lit_detr.model.eval().to(device)

# Define the metric used in evaluation process
ap_metrics = alonet.metrics.ApMetrics()

# Make all predictions to use metric defined
for it, data in enumerate(coco_loader.val_dataloader()):
    frame = aloscene.Frame.batch_list(data).to(device)

    pred_boxes = lit_detr.inference(lit_detr(frame))[0]
    gt_boxes = frame.boxes2d[0]  # Get gt boxes as BoundingBoxes2D.

    ap_metrics.add_sample(pred_boxes, gt_boxes)  # Add samples to evaluate metrics

    print(f"it:{it}", end="\r")
    if args.ap_limit is not None and it > args.ap_limit:
        break

# Show the results
print("Total eval batch:", it)
ap_metrics.calc_map(print_result=True)
