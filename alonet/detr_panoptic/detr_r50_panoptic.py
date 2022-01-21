"""Module to create a :mod:`PanopticHead <alonet.detr_panoptic.detr_panoptic>` model, using
:mod:`DetrR50 <alonet.detr.detr_r50>` as detection architecture.
"""

from argparse import Namespace
from alonet.detr_panoptic import PanopticHead
from alonet.detr import DetrR50


class DetrR50Panoptic(PanopticHead):
    """:mod:`DetrR50 <alonet.detr.detr_r50>` + :mod:`PanopticHead <alonet.detr_panoptic.detr_panoptic>`

    Parameters
    ----------
    num_classes : int, optional
        Number of classes in the :attr:`class_embed` output layer, by default 250
    background_class : int, optional
        Background class, by default None
    detr_weights : str, optional
        Load weights to :mod:`DetrR50 <alonet.detr.detr_r50>`, by default None
    weights : str, optional
        Load weights from path or model_name, by default None
    **kwargs
        Initial parameters of :mod:`PanopticHead <alonet.detr_panoptic.detr_panoptic>` module

    Raises
    ------
    ValueError
        :attr:`weights` must be a '.pth' or '.ckpt' file
    """

    def __init__(
        self,
        num_classes: int = 250,
        background_class: int = None,
        detr_weights: str = None,
        weights: str = None,
        *args: Namespace,
        **kwargs: dict,
    ):
        """Init method"""
        base_model = DetrR50(num_classes=num_classes, background_class=background_class, weights=detr_weights)
        super().__init__(*args, DETR_module=base_model, weights=weights, **kwargs)


def main(image_path):
    import torch
    import time
    import aloscene

    device = torch.device("cuda")

    # Load model
    model = DetrR50Panoptic(weights="detr-r50-panoptic")
    model.to(device).eval()

    # Open and prepare a batch for the model
    frame = aloscene.Frame(image_path).norm_resnet()
    frames = aloscene.Frame.batch_list([frame])
    frames = frames.to(device)

    # GPU warm up
    [model(frames) for _ in range(3)]

    tic = time.time()
    with torch.no_grad():
        [model(frames) for _ in range(20)]
    toc = time.time()
    print(f"{(toc - tic)/20*1000} ms")

    # Predict boxes/masks
    m_outputs = model(frames)  # Pred of size (B, NQ, H//4, W//4)
    pred_boxes, pred_masks = model.inference(m_outputs, frame_size=frames.HW, threshold=0.85)

    # Add and display the boxes/masks predicted
    frame.append_boxes2d(pred_boxes[0], "pred_boxes")
    frame.append_segmentation(pred_masks[0], "pred_masks")
    frame.get_view().render()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Detr R50 Panoptic inference on image")
    parser.add_argument("image_path", type=str, help="Path to the image for inference")
    args = parser.parse_args()
    main(args.image_path)
