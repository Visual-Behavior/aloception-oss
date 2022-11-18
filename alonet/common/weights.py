import torch
import requests
import os
from alonet.common.pl_helpers import vb_folder, checkpoint_handler

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
    ]
}


def load_weights(
        model,
        weights=None,
        run_id=None,
        project_run_id=None,
        checkpoint="best",
        monitor="val_loss",
        device=torch.device("cpu"),
        strict_load_weights=True,
    ):
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
    assert run_id is not None or weights is not None, "run_id or weights must be set."

    if weights is None:
        if project_run_id is None:
            Exception(
                "project_run_id need to be set if we load model from run_id."
            )
        run_id_project_dir = os.path.join(vb_folder(), f"project_{project_run_id}", run_id)
        ckpt_path = checkpoint_handler(checkpoint, run_id_project_dir, monitor)
        weights = os.path.join(run_id_project_dir, ckpt_path)
        if not os.path.exists(weights):
            raise Exception(f"Impossible to load the ckpt at the following destination:{weights}")
        print(f"Loading ckpt from {run_id} at {weights}")

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
        model.load_state_dict(checkpoint, strict=strict_load_weights)
    else:
        raise Exception(f"Cant load the weights: {weights}")
