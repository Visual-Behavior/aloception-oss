import torchvision
import torch
from torchvision.io.image import ImageReadMode

from aloscene.io.utils.errors import InvalidSampleError


def load_image(image_path):
    """
    Load an image with pytorch in float32 format

    Parameters
    ----------
    image_path : str
        path of the image

    Returns
    -------
    image : torch.Tensor
        tensor containing the image
    """
    try:
        image = torchvision.io.read_image(image_path, ImageReadMode.RGB).type(torch.float32)
    except RuntimeError as e:
        raise InvalidSampleError(f"[Alodataset Warning] Invalid image: {image_path} error={e}")
    return image
