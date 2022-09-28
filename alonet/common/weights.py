import torch
import requests
import os

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


def vb_fodler():
    home = os.getenv("HOME")
    alofolder = os.path.join(home, ".aloception")
    if not os.path.exists(alofolder):
        os.mkdir(alofolder)
    return alofolder


def load_weights(model, weights, device, strict_load_weights=True):
    """Load and/or download weights from public cloud

    Parameters
    ----------
    model: torch.model
        The torch model to load the weights into
    weights: str
        Weights names. Must be set into WEIGHT_NAME_TO_FILES
    device: torch.device
        Device to load the weights into
    """
    weights_dir = os.path.join(vb_fodler(), "weights")

    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    if os.path.splitext(weights.lower())[1] == ".pth":
        checkpoint = torch.load(weights, map_location=device)
        if "model" in checkpoint:
            checkpoint = checkpoint["model"]
        model.load_state_dict(checkpoint, strict=strict_load_weights)
        print(f"Weights loaded from {weights}")
    elif os.path.splitext(weights.lower())[1] == ".ckpt":
        checkpoint = torch.load(weights, map_location=device)["state_dict"]
        checkpoint = {k.replace("model.", "") if "model." in k else k: v for k, v in checkpoint.items()}
        model.load_state_dict(checkpoint, strict=strict_load_weights)
        print(f"Weights loaded from {weights}")
    elif weights in WEIGHT_NAME_TO_FILES:
        weights_dir = os.path.join(weights_dir, weights)
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
        model.load_state_dict(checkpoint, strict=strict_load_weights)
    else:
        raise Exception(f"Cant load the weights: {weights}")
