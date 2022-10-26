import numpy as np


def load_depth(path):
    """
    Load Depth data

    Parameters
    ----------
    path: str
        path to the disparity file. Supported format: {".npy"} or {".npz"}. If your file is stored differently, as an
        alternative, you can open the file yourself and then create the Depth augmented Tensor from the depth
        data.
    """
    if path.endswith(".npy"):
        return np.load(path)
    if path.endswith(".npz"):
        with np.load(path) as f:
            keys = list(f.keys())
            if len(keys) > 1:
                raise ValueError(f"Cannot load depth from {path} containing {len(keys)} > 1 arrays.")
            return f[keys[0]][None, ...].astype(np.float32) / 100
    else:
        raise ValueError(
            f"Unknown extension for depth file: {path}. As an alternative you can load the file manually\
            and then create the Depth augmented tensor from the depth data."
        )
