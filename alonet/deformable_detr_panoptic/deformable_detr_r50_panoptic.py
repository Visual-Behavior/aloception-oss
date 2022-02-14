"""Module to create a :mod:`PanopticHead <alonet.detr_panoptic.detr_panoptic>` model, using
:mod:`DeformableDetrR50 <alonet.deformable_detr.deformable_detr_r50>` as detection architecture.
"""

from argparse import Namespace

from alonet.common import load_weights
from alonet.detr_panoptic import PanopticHead
from alonet.deformable_detr import DeformableDetrR50, DeformableDetrR50Refinement


class DeformableDetrR50Panoptic(PanopticHead):
    """:mod:`DeformableDetrR50 <alonet.detr_deformable.deformable_detr_r50>` +
    :mod:`PanopticHead <alonet.detr_panoptic.detr_panoptic>`

    Parameters
    ----------
    num_classes : int, optional
        Number of classes in the :attr:`class_embed` output layer, by default 250
    activation_fn : str, optional
        Activation function for classification head. Either ``sigmoid`` or ``softmax``, by default "sigmoid".
    with_box_refine : bool, optional
        Use iterative box refinement, see paper for more details, by default False
    deformable_weights : str, optional
        Load weights to :mod:`DeformableDetrR50 <alonet.detr_deformable.deformable_detr_r50>`, by default None
    weights : str, optional
        Load weights from path or model_name, by default None
    strict_load_weights : bool
        Load the weights (if any given) with strict = ``True`` (by default)
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
        activation_fn: str = "sigmoid",
        with_box_refine: bool = False,
        return_intermediate_dec: bool = True,
        deformable_weights: str = None,
        weights: str = None,
        strict_load_weights: bool = True,
        *args: Namespace,
        **kwargs: dict,
    ):
        """Init method"""
        base_model = DeformableDetrR50Refinement if with_box_refine else DeformableDetrR50
        base_model = base_model(
            num_classes=num_classes,
            weights=deformable_weights,
            activation_fn=activation_fn,
            return_intermediate_dec=return_intermediate_dec,
        )
        super().__init__(*args, DETR_module=base_model, weights=None, **kwargs)

        # Load weights
        list_weights = ["deformable-detr-r50-panoptic", "deformable-detr-r50-refinement-panoptic"]
        if weights is not None:
            if ".pth" in weights or ".ckpt" in weights or weights in list_weights:
                load_weights(self, weights, self.device, strict_load_weights=strict_load_weights)
            else:
                raise ValueError(f"Unknown weights: '{weights}'")


def main(image_path):
    import torch
    import time
    import aloscene

    device = torch.device("cuda")

    # Load model
    model = DeformableDetrR50Panoptic(weights="deformable-detr-r50-panoptic")
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
