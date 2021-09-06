import torch

flip_x_axis_mat = torch.Tensor([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def get_x_rotation_matrixes(theta: torch.Tensor) -> torch.Tensor:
    """Get rotation matrices by X axis

    Parameters
    ----------
    theta : torch.Tensor
        rotation angle in rad, shape (n, )

    Returns
    -------
    torch.Tensor
        rotation matrices, shape (n, 3, 3)
    """
    assert len(theta.shape) == 1
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    zeros = torch.zeros(theta.shape)
    ones = torch.ones(theta.shape)
    rot = torch.stack([ones, zeros, zeros, zeros, cos_theta, -sin_theta, zeros, sin_theta, cos_theta], dim=1)  # (n, 9)
    return rot.reshape((-1, 3, 3))


def get_y_rotation_matrixes(theta: torch.Tensor) -> torch.Tensor:
    """Get rotation matrices by Y axis

    Parameters
    ----------
    theta : torch.Tensor
        rotation angle in rad, shape (n, )

    Returns
    -------
    torch.Tensor
        rotation matrices, shape (n, 3, 3)
    """
    assert len(theta.shape) == 1
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    zeros = torch.zeros(theta.shape).to(theta.device)
    ones = torch.ones(theta.shape).to(theta.device)
    rot = torch.stack([cos_theta, zeros, sin_theta, zeros, ones, zeros, -sin_theta, zeros, cos_theta], dim=1)  # (n, 9)
    return rot.reshape((-1, 3, 3))


def get_z_rotation_matrixes(theta: torch.Tensor) -> torch.Tensor:
    """Get rotation matrices by Z axis

    Parameters
    ----------
    theta : torch.Tensor
        rotation angle in rad, shape (n, )

    Returns
    -------
    torch.Tensor
        rotation matrices, shape (n, 3, 3)
    """
    assert len(theta.shape) == 1
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    zeros = torch.zeros(theta.shape)
    ones = torch.ones(theta.shape)
    rot = torch.stack([cos_theta, -sin_theta, zeros, sin_theta, cos_theta, zeros, zeros, zeros, ones], dim=1)  # (n, 9)
    return rot.reshape((-1, 3, 3))


@torch.no_grad()
def is_rotation_matrix(R: torch.Tensor) -> bool:
    """
    Check if a matrix is a valid rotation matrix

    Parameters:
    ----------
    R: torch.Tensor, (3, 3), rotation matrix in question

    Returns
    -------
    bool
        True if a valid rotation matrix
    """
    if not isinstance(R, torch.Tensor):
        R = R.clone().as_tensor()
    Rt = torch.transpose(R, 0, 1)
    shouldBeIdentity = torch.matmul(Rt, R)
    identical = torch.eye(3, dtype=R.dtype, device=R.device)
    n = torch.linalg.norm(identical - shouldBeIdentity)
    return n < 1e-6


def rotation_matrix_to_euler_angles(R: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix to euler angles for each axis (X, Y, Z)

    Parameters
    ----------
    R : torch.Tensor
        rotation matrix, (3, 3)

    Returns
    -------
    torch.Tensor
        rotation angle for X, Y, Z axis respectively
    """
    assert is_rotation_matrix(R)
    sy = torch.sqrt(R[2, 1] * R[2, 1] + R[2, 2] * R[2, 2])
    x = torch.atan2(R[2, 1], R[2, 2])
    y = torch.atan2(-R[2, 0], sy)
    z = torch.atan2(R[1, 0], R[0, 0])
    return torch.stack([x, y, z])
