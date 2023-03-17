import numpy as np


def load_depth(path):
    """
    Load Depth data

    Parameters
    ----------
    path: str
        path to the disparity file. Supported format: {".npy",".npz"}. If your file is stored differently, as an
        alternative, you can open the file yourself and then create the Depth augmented Tensor from the depth
        data.
    """
    if path.endswith(".npy"):
        return np.load(path)
    elif path.endswith(".npz"):
        return np.load(path)["arr_0"]
    else:
        raise ValueError(
            f"Unknown extension for depth file: {path}. As an alternative you can load the file manually\
            and then create the Depth augmented tensor from the depth data."
        )
