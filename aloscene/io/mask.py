import torch
import torchvision
from aloscene.io.utils.errors import InvalidSampleError
from torchvision.io.image import ImageReadMode

def load_mask_png(path):
    try:
        image = torchvision.io.read_image(path, ImageReadMode.GRAY).type(torch.float32) / 255.0
    except RuntimeError as e:
        raise InvalidSampleError(f"[Alodataset Warning] Invalid mask file: {path}")
    return image


def load_mask(path):
    if path.endswith(".zfd"):
        raise Exception("zfd format is not supported.")
    elif path.lower().endswith(".png"):
        return load_mask_png(path)
    else:
        raise ValueError()
