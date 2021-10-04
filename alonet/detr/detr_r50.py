import time
import torchvision
import argparse
import torch

from alonet.detr import Detr
import aloscene
import alonet
from alonet.detr.misc import assert_and_export_onnx


class DetrR50(Detr):
    """DETR R50 as described in the paper: https://arxiv.org/abs/2005.12872"""

    def __init__(self, *args, num_classes=91, background_class=91, **kwargs):
        # Positional encoding
        position_embedding = self.build_positional_encoding(hidden_dim=256, position_embedding="sin")
        # Backbone
        backbone = self.build_backbone("resnet50", train_backbone=True, return_interm_layers=True, dilation=False,)
        num_channels = backbone.num_channels
        backbone = alonet.detr.backbone.Joiner(backbone, position_embedding)
        backbone.num_channels = num_channels
        # Build transformer
        transformer = self.build_transformer(
            hidden_dim=256,
            dropout=0.1,
            nheads=8,
            dim_feedforward=2048,
            num_encoder_layers=6,
            num_decoder_layers=6,
            normalize_before=False,
        )
        super().__init__(
            backbone,
            transformer,
            *args,
            num_classes=num_classes,
            num_queries=100,
            background_class=background_class,
            **kwargs,
        )


def main(image_path):
    device = torch.device("cuda")

    # Load model
    model = DetrR50(num_classes=91, weights="detr-r50", device=device).eval()

    # Open and prepare a batch for the model
    frame = aloscene.Frame(image_path).norm_resnet()
    frames = aloscene.Frame.batch_list([frame])
    frames = frames.to(device)

    with torch.no_grad():
        # Measure inference time
        tic = time.time()
        [model(frames) for _ in range(20)]
        toc = time.time()
        print(f"{(toc - tic)/20*1000} ms")

        # Predict boxes
        m_outputs = model(frames)
    pred_boxes = model.inference(m_outputs)

    # Add and display the predicted boxes
    frame.append_boxes2d(pred_boxes[0], "pred_boxes")
    frame.get_view([frame.boxes2d]).add(frame.get_view([frame])).render()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detr R50 inference on image")
    parser.add_argument("image_path", type=str, help="Path to the image for inference")
    args = parser.parse_args()
    main(args.image_path)
