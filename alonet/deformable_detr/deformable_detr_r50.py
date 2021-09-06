import argparse
import torch

import alonet
from alonet.deformable_detr import DeformableDETR
from alonet.deformable_detr.backbone import Joiner
import aloscene


class DeformableDetrR50(DeformableDETR):
    """Deformable Detr with Resnet50 backbone"""

    def __init__(self, *args, return_intermediate_dec: bool = True, num_classes=91, **kwargs):
        backbone = self.build_backbone(
            backbone_name="resnet50", train_backbone=True, return_interm_layers=True, dilation=False
        )
        position_embed = self.build_positional_encoding(hidden_dim=256)
        backbone = Joiner(backbone, position_embed)
        transformer = self.build_transformer(
            hidden_dim=256,
            dropout=0.1,
            nheads=8,
            dim_feedforward=1024,
            enc_layers=6,
            dec_layers=6,
            num_feature_levels=4,
            dec_n_points=4,
            enc_n_points=4,
            return_intermediate_dec=return_intermediate_dec,
        )

        super().__init__(backbone, transformer, *args, num_classes=num_classes, with_box_refine=False, **kwargs)


if __name__ == "__main__":
    from alonet.deformable_detr import DeformableDetrR50

    # For testing
    device = torch.device("cuda")
    parser = argparse.ArgumentParser(description="Detr R50 inference on image")
    parser.add_argument("image_path", type=str, help="Path to the image for inference")
    args = parser.parse_args()

    model = DeformableDetrR50(weights="deformable-detr-r50").eval()

    image_path = args.image_path
    frame = aloscene.Frame(image_path).norm_resnet()
    frames = aloscene.Frame.batch_list([frame])
    frames = frames.to(device)

    m_outputs = model(frames)
    pred_boxes = model.inference(m_outputs)
    # Add and display the predicted boxes
    frame.append_boxes2d(pred_boxes[0], "pred_boxes")
    frame.get_view([frame.boxes2d]).render()
