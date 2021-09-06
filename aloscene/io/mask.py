import numpy as np
import torch
import torchvision
from aloscene.io.utils.errors import InvalidSampleError



def load_mask_png(path):
    try:
        image = torchvision.io.read_image(path).type(torch.float32) / 255.0
    except RuntimeError as e:
        raise InvalidSampleError(f"[Alodataset Warning] Invalid mask file: {path}")
    return image


def load_mask(path):
    if path.endswith(".zfd"):
        raise Exception("zfd format is not supported.")
    elif path.endswith(".png"):
        return load_mask_png(path)
    else:
        raise ValueError()
