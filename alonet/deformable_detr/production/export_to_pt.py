import argparse
import torch

import alonet
from alonet.deformable_detr import DeformableDETR
from alonet.deformable_detr.backbone import Joiner
from alonet.deformable_detr.ops.functions import load_ops
import aloscene

if __name__ == "__main__":
    from alonet.deformable_detr import DeformableDetrR50

    # For testing
    device = torch.device("cuda")

    model = DeformableDetrR50(weights="deformable-detr-r50", tracing=True, aux_loss=False).eval()

    example_input = torch.rand(1, 4, 800, 1333).to(device)
    example_input2 = torch.rand(1, 4, 400, 600).to(device)
    traced_script_module = torch.jit.trace(model, example_input, check_inputs=[example_input, example_input2])

    traced_script_module.save("deformable-detr-r50.pt")

    print("traced_script_module", traced_script_module)
