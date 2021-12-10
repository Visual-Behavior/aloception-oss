import torchvision
import torch
from torchvision.io.image import ImageReadMode

from aloscene.io.utils.errors import InvalidSampleError
import cv2 as cv


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
        image = cv.imread(image_path)
    except RuntimeError as e:
        raise InvalidSampleError(f"[Alodataset Warning] Invalid image: {image_path} error={e}")
    image = torch.tensor(cv.cvtColor(image, cv.COLOR_BGR2RGB), dtype=torch.float32).permute(2, 0, 1)
    return image
