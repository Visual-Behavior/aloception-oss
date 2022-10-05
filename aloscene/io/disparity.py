from PIL import Image
import numpy as np
import torch
import re


def load_pfm_np(path, flip=True, clean=False):
    with open(path, "rb") as file:
        color = None
        width = None
        height = None
        scale = None
        endian = None
        header = file.readline().rstrip().decode("ascii")
        # print("--" + header + "--")
        if header == "PF":
            color = True
        elif header == "Pf":
            # print("Here!")
            color = False
        else:
            raise Exception("Not a PFM file.")

        dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("ascii"))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception("Malformed PFM header.")

        scale = float(file.readline().rstrip())
        if scale < 0:  # little-endian
            endian = "<"
            scale = -scale
        else:
            endian = ">"  # big-endian
        data = np.fromfile(file, endian + "f")
        shape = (height, width, 3) if color else (height, width, 1)
        data = np.reshape(data, shape)
        if flip:
            data = np.flip(data, 0)

        if clean:
            data[data == np.nan] = 0.0
            data[data < -1e30] = 0.0
            data[data > 1e30] = 0.0

    return np.squeeze(data, axis=-1) if color and data.shape[-1] == 1 else data


def load_disp_pfm(path, flip=True, clean=False):
    """
    Loads disparity as a torch.Tensor

    Parameters
    ----------
    flip : bool
        flip the vertical axis. Default is True
    clean : bool
        changes nan and extreme values to zero. Default is False
    add_temporal :
        adds a temporal dimension as first channel.

    Returns
    -------
    disp : torch.Tensor
        disparity map
    """
    disp_np = load_pfm_np(path, flip, clean)
    disp_np = disp_np.transpose([2, 0, 1]).astype(np.float32)  # pytorch convention : C, H, W
    disp = torch.from_numpy(disp_np)
    return disp


def load_disp_png(path, decoding="rgb", scale=256.0, negate=False):
    """Return disparity read from filename."""
    f_in = np.array(Image.open(path))
    if decoding == "rgb":
        d_r = f_in[:, :, 0].astype("float64")
        d_g = f_in[:, :, 1].astype("float64")
        d_b = f_in[:, :, 2].astype("float64")
        disp = d_r * 4 + d_g / (2 ** 6) + d_b / (2 ** 14)
    elif decoding == "uint16":  # Kitti_stereo
        disp = f_in.astype("float32") / scale
    disp = np.expand_dims(disp, axis=-1)
    if negate:
        disp = -1 * disp
    # from numpy to pytorch
    disp = disp.transpose([2, 0, 1]).astype(np.float32)
    disp = torch.from_numpy(disp)
    return disp


def load_disp(path, png_negate=None):
    """
    Load disparity

    Parameters
    ----------
    path: str
        path to the disparity file. Supported format: {".pfm", ".png"}
    png_negate: bool
        if true, the sign of disparity is reversed
        this parameter should be explicitely set every time a .png file is used.
    """
    if path.endswith(".pfm"):
        return load_disp_pfm(path)
    elif path.endswith(".zfd"):
        raise Exception(".zfd format is not supported.")
    elif path.endswith(".png") and png_negate is None:
        raise NotImplementedError(
            "When loading a disparity PNG file, you must set the value of the `png_negate` parameter."
        )
    elif path.endswith(".png"):
        return load_disp_png(path, negate=png_negate)
    else:
        raise ValueError(f"Unknown extension for disparity file: {path}")
