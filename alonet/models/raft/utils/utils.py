import torch
import torch.nn.functional as F


def bilinear_sampler(img, coords, mode="bilinear", mask=False):
    """Wrapper for grid_sample, uses pixel coordinates"""
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def upflow8(flow, mode="bilinear"):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return 8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


class Padder:
    """Pad spatial dims to multiples of 8"""

    def __init__(self):
        pass

    def _set_pad(self, frame):
        self.h, self.w = frame.HW
        self.pad_h = (((self.h // 8) + 1) * 8 - self.h) % 8
        self.pad_w = (((self.h // 8) + 1) * 8 - self.w) % 8
        # padding offsets
        self.top = self.pad_h // 2
        self.bottom = self.pad_h - self.top
        self.left = self.pad_w // 2
        self.right = self.pad_w - self.left

    def pad(self, frame):
        """Pad frame but not its labels"""
        self._set_pad(frame)
        _pad = [self.left, self.right, self.top, self.bottom]
        frame.rename_(None, auto_restore_names=True)  # temporarily remove tensor names
        frame = F.pad(frame, _pad, mode="replicate")  # because F.pad does not support named tensors
        return frame

    def unpad(self, tensor):
        h, w = tensor.shape[-2:]
        top, bottom, left, right = self.top, self.bottom, self.left, self.right
        return tensor[..., top : h - bottom, left : w - right]
