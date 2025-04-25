import torch
import requests
import os
from typing import Optional
from safetensors.torch import load_file

from alonet.common.helpers import vb_folder


WEIGHT_NAME_TO_FILES = {
    "detr-r50": ["https://storage.googleapis.com/visualbehavior-publicweights/detr-r50/detr-r50.pth"],
    "deformable-detr-r50": [
        "https://storage.googleapis.com/visualbehavior-publicweights/detr-r50-deformable/deformable-detr-r50.pth"
    ],
    "deformable-detr-r50-refinement": [
        "https://storage.googleapis.com/visualbehavior-publicweights/detr-r50-deformable-refinement/deformable-detr-r50-refinement.pth"
    ],
    "raft-things": ["https://storage.googleapis.com/visualbehavior-publicweights/raft-things/raft-things.pth"],
    "raft-chairs": ["https://storage.googleapis.com/visualbehavior-publicweights/raft-chairs/raft-chairs.pth"],
    "raft-sintel": ["https://storage.googleapis.com/visualbehavior-publicweights/raft-sintel/raft-sintel.pth"],
    "raft-small": ["https://storage.googleapis.com/visualbehavior-publicweights/raft-small/raft-small.pth"],
    "raft-kitti": ["https://storage.googleapis.com/visualbehavior-publicweights/raft-kitti/raft-kitti.pth"],
    "trackformer-deformable-mot": [
        "https://storage.googleapis.com/visualbehavior-publicweights/trackformer-deformable-mot/trackformer-deformable-mot.pth"
    ],
    "trackformer-crowdhuman-deformable-mot": [
        "https://storage.googleapis.com/visualbehavior-publicweights/trackformer-crowdhuman-deformable-mot/trackformer-crowdhuman-deformable-mot.pth"
    ],
    "detr-r50-panoptic": [
        "https://storage.googleapis.com/visualbehavior-publicweights/detr-r50-panoptic/detr-r50-panoptic.pth"
    ],
    "detr-r50-things-stuffs": [
        "https://storage.googleapis.com/visualbehavior-publicweights/detr-r50-things-stuffs/detr-r50-things-stuffs.pth"
    ],
}


def load_weights(
    model: torch.nn.Module,
    weights: str,
    prefix_to_remove: Optional[str] = None,
    device: torch.device = torch.device("cpu"),
    strict_load_weights: bool = True,
) -> None:
    """Load and/or download weights from public cloud

    Parameters
    ----------
    model: torch.nn.Module
        The torch model to load the weights into
    weights: str
        Weights names. Must be set into WEIGHT_NAME_TO_FILES
    prefix_to_remove: Optional[str]
        Prefix to remove from the weights keys
    device: torch.device
        Device to load the weights into
    strict_load_weights: bool
        If True, the weights are loaded with strict=True
    """
    if os.path.splitext(weights.lower())[1] == ".pth":
        checkpoint = torch.load(weights, map_location=device)
        if "model" in checkpoint:
            checkpoint = checkpoint["model"]
    elif os.path.splitext(weights.lower())[1] == ".ckpt":
        checkpoint = torch.load(weights, map_location=device)["state_dict"]
    elif os.path.splitext(weights.lower())[1] == ".safetensors":
        if device == torch.device("cpu"):
            device_str = "cpu"
        else:
            device_str = "cuda"
        checkpoint = load_file(weights, device=device_str)
    elif weights in WEIGHT_NAME_TO_FILES:
        weights_dir = os.path.join(vb_folder(create_if_not_found=True), "weights", weights)
        if not os.path.exists(weights_dir):
            os.makedirs(weights_dir)
        for f in WEIGHT_NAME_TO_FILES[weights]:
            fname = f.split("/")[-1]
            if not os.path.exists(os.path.join(weights_dir, fname)):
                print("Download....", f)
                r = requests.get(f, allow_redirects=True)
                open(os.path.join(weights_dir, fname), "wb").write(r.content)

        wfile = os.path.join(weights_dir, f"{weights}.pth")
        print("Load weights from", wfile)
        checkpoint = torch.load(wfile, map_location=device)
        checkpoint = checkpoint["model"] if "model" in checkpoint else checkpoint
    else:
        raise Exception(f"Cant load the weights: {weights}")

    processed_checkpoint = {}
    # Remove prefix if provided
    if prefix_to_remove is not None:
        for k, v in checkpoint.items():
            if k.startswith(prefix_to_remove):
                processed_checkpoint[k[len(prefix_to_remove) :]] = v
            else:
                processed_checkpoint[k] = v
    else:
        processed_checkpoint = checkpoint

    model.load_state_dict(processed_checkpoint, strict=strict_load_weights)
    print(f"Weights loaded from {weights}")
