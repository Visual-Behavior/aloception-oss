import torch
from aloscene.tensors import AugmentedTensor


def coords2rtheta(K, size, distortion, projection="pinhole"):
    """Compute r_d and theta from image coordinates.

    Parameters
    ----------
    K: aloscene.CameraIntrinsic
        Intrinsic matrix of camera.
    size: tuple
        (H, W) height and width of image
    distortion: float
        Distortion coefficient using for wide angle camera.
    projection: str
        Projection model: Only pinhole and equidistant projection are supported.
    """
    h, w = size
    focal = K.focal_length[..., 0]
    principal_point = K.principal_points[..., :][:, None, None]

    coords = torch.meshgrid(torch.arange(h), torch.arange(w))
    coords = torch.stack(coords[::-1], dim=0).float().to(K.device)
    coords = coords - principal_point
    r_d = coords[:2, ...] * coords[:2, ...]
    r_d = torch.sqrt(torch.sum(r_d, dim=0, keepdim=True))

    if projection == "pinhole":
        theta = torch.atan(r_d / focal)
    elif projection == "equidistant":
        theta = r_d / (focal * distortion)
    else:
        raise NotImplementedError

    theta = AugmentedTensor(theta, names=("C", "H", "W"))
    r_d = AugmentedTensor(r_d, names=("C", "H", "W"))

    return r_d, theta
