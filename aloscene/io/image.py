from aloscene.io.utils.errors import InvalidSampleError

import cv2
import torch
import torchvision
import numpy as np
from torchvision.io.image import ImageReadMode


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
        pass
    try:
        image = cv2.imread(image_path).astype(np.float32)
        image = np.moveaxis(image, 2, 0)
    except RuntimeError as e:
        raise InvalidSampleError(f"[Alodataset Warning] Invalid image: {image_path} error={e}")
    return image
