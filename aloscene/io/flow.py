import numpy as np
import torch



def load_flow_flo(flo_path):
    """
    Load a 2D flow map with pytorch in float32 format

    Parameters
    ----------
    flo_path : str
        path of the ".flo" file
    add_temporal : bool, default=True
        add a first dimension for time

    Returns
    -------
    flow : torch.Tensor
        tensor containing the flow map
    """
    with open(flo_path, "rb") as f:
        header = f.read(4)
        if header.decode("utf-8") != "PIEH":
            raise Exception("Flow file header does not contain PIEH")
        width = np.fromfile(f, np.int32, 1).squeeze()
        height = np.fromfile(f, np.int32, 1).squeeze()
        flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))
    flow = flow.transpose([2, 0, 1])  # pytorch convention : C, H, W
    flow = torch.from_numpy(flow)
    return flow




def load_flow(flow_path):
    if flow_path.endswith(".flo"):
        return load_flow_flo(flow_path)
    elif flow_path.endswith(".zfd"):
        raise Exception("zfd format is not supported.")
    else :
        raise ValueError(f"Unknown extension for flow file: {flow_path}")
