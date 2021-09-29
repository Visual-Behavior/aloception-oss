import argparse
import torch

import alonet
from alonet.deformable_detr import DeformableDETR
from alonet.deformable_detr.backbone import Joiner
import aloscene


if __name__ == "__main__":
    from alonet.deformable_detr import DeformableDetrR50

    # For testing
    device = torch.device("cuda")

    model = DeformableDetrR50(weights="deformable-detr-r50", tracing=True, aux_loss=False).eval()

    example_input = torch.rand(1, 4, 800, 1333).to(device)
    traced_script_module = torch.jit.trace(model, example_input)

    traced_script_module.save("deformable-detr-r50.pt")

    print("traced_script_module", traced_script_module)

    # image_path = args.image_path
    # frame = aloscene.Frame(image_path).norm_resnet()
    # frames = aloscene.Frame.batch_list([frame])
    # frames = frames.to(device)

    # m_outputs = model(frames)
    # pred_boxes = model.inference(m_outputs)
    ## Add and display the predicted boxes
    # frame.append_boxes2d(pred_boxes[0], "pred_boxes")
    # frame.get_view([frame.boxes2d]).render()
