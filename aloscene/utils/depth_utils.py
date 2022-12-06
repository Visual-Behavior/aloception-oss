from aloscene.tensors import AugmentedTensor
from typing import Union, Tuple, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch


def coords2rtheta(
    K, size: Tuple[int, int], distortion: Union[float, Tuple[float, float]], projection: str = "pinhole"
):
    """Compute r_d and theta from image coordinates.

    Parameters
    ----------
    K: aloscene.CameraIntrinsic
        Intrinsic matrix of camera.
    size: Tuple[int, int]
        (H, W) height and width of image
    distortion: Union[float, Tuple[float, float]]
        Distortion coefficient(s) for wide angle cameras.
    projection: str
        Projection model: Only pinhole, equidistant and kumler_bauer projections are supported.
    """
    h, w = size
    focal = K.focal_length[..., 0]
    principal_point = K.principal_points[..., :]
    valid_names = []
    for name in K.names:
        if name in {"B", "T"}:
            valid_names.append(name)
            focal = focal[0, ...]
            principal_point = principal_point[0, ...]
    principal_point = principal_point[:, None, None]

    coords = torch.meshgrid(torch.arange(h), torch.arange(w))
    coords = torch.stack(coords[::-1], dim=0).float().to(K.device)
    coords = coords - principal_point
    r_d = coords[:2, ...] * coords[:2, ...]
    r_d = torch.sqrt(torch.sum(r_d, dim=0, keepdim=True))
    for _ in range(len(valid_names)):
        r_d = r_d.unsqueeze(0)

    if projection == "pinhole":
        theta = torch.atan(r_d / focal)
    elif projection == "equidistant":
        dist_coef = distortion[0] if isinstance(distortion, (Sequence, torch.Tensor)) else distortion
        theta = r_d / (focal * dist_coef)
    elif projection == "kumler_bauer":
        assert isinstance(
            distortion, (list, torch.Tensor)
        ), f"Kumler-Bauer projection needs to be Sequence or torch.Tensor. Found {type(distortion)}"
        if isinstance(distortion, torch.Tensor):
            k1, k2 = distortion[..., 0, None, None, None], distortion[..., 1, None, None, None]
        else:
            k1, k2 = distortion[0], distortion[1]
        theta = torch.arcsin(r_d / k1) / k2
    else:
        raise NotImplementedError

    theta = AugmentedTensor(theta, names=valid_names + ["C", "H", "W"])
    r_d = AugmentedTensor(r_d, names=valid_names + ["C", "H", "W"])
    return r_d, theta


def add_colorbar(data, vmin, vmax, colormap):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    pos = plt.imshow(data, cmap=colormap, interpolation="none")
    axins1 = ax.inset_axes([0.8, 0.2, 0.04, 0.6])
    plt.axis("off")
    fig.colorbar(pos, cax=axins1, orientation="vertical")
    pos.set_clim(vmin=vmin, vmax=vmax)
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,)) / 255.0
    plt.close(fig)

    return data