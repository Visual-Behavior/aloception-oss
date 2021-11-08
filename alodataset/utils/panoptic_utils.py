from typing import Union
import numpy as np
import torch

# from alonet.metrics.compute_pq import VOID

VOID_CLASS_ID = -1
GLOBAL_COLOR_SET = np.random.uniform(0, 1, (300, 3))
GLOBAL_COLOR_SET[VOID_CLASS_ID] = [0, 0, 0]
OFFSET = 256 * 256 * 256


# Function get from PanopticAPI: https://github.com/cocodataset/panopticapi/blob/master/panopticapi/utils.py
def rgb2id(color: Union[list, np.ndarray]):
    if isinstance(color, np.ndarray) and len(color.shape) == 3:
        if color.dtype == np.uint8:
            color = color.astype(np.int32)
        return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
    return int(color[0] + 256 * color[1] + 256 * 256 * color[2])


# Function get from PanopticAPI: https://github.com/cocodataset/panopticapi/blob/master/panopticapi/utils.py
def id2rgb(id_map: np.ndarray, random_color: bool = True):
    if random_color:
        return GLOBAL_COLOR_SET[id_map]
    if isinstance(id_map, np.ndarray):
        id_map_copy = id_map.copy()
        rgb_shape = tuple(list(id_map.shape) + [3])
        rgb_map = np.zeros(rgb_shape, dtype=np.uint8)
        for i in range(3):
            rgb_map[..., i] = id_map_copy % 256
            id_map_copy //= 256
        return rgb_map / 255.0
    color = []
    for _ in range(3):
        color.append((id_map % 256) / 255.0)
        id_map //= 256
    return color


# Function get from DETR:
# https://github.com/facebookresearch/detr/blob/fe752a89b352284c7395dbb629bedaa64271b0f4/util/box_ops.py#L64
def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks
    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.
    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = masks * x.unsqueeze(0)
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = masks * y.unsqueeze(0)
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)
