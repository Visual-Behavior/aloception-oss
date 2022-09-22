import numpy as np
from typing import Union


# https://github.com/utiasSTARS/pykitti/tree/master
def read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, "r") as f:
        for line in f.readlines():
            if line == "\n":
                continue
            key, value = line.split(" ", 1)
            key = key.strip(":")
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data


def transform_from_rot_trans(R, t):
    """Transforation matrix from rotation matrix and translation vector."""
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))


def load_calib_rigid(filepath):
    """Read a rigid transform calibration file as a numpy.array."""
    data = read_calib_file(filepath)
    return transform_from_rot_trans(data["R"], data["T"])


def load_calib_cam_to_cam(cam_to_cam_filepath, velo_to_cam_file: Union[str, None] = None):
    # We'll return the camera calibration as a dictionary
    data = {}

    # Load and parse the cam-to-cam calibration data
    filedata = read_calib_file(cam_to_cam_filepath)

    names = ["P_rect_00", "P_rect_01", "P_rect_02", "P_rect_03"]
    if "P0" in filedata:
        names = ["P0", "P1", "P2", "P3"]

    # Create 3x4 projection matrices
    p_rect = [np.reshape(filedata[p], (3, 4)) for p in names]

    for i, p in enumerate(p_rect):
        data[f"P_rect_{i}0"] = p

    # Compute the rectified extrinsics from cam0 to camN
    rectified_extrinsics = [np.eye(4) for _ in range(4)]
    for i in range(4):
        rectified_extrinsics[i][0, 3] = p_rect[i][0, 3] / p_rect[i][0, 0]
        data[f"T_cam{i}_rect"] = rectified_extrinsics[i]

        # Compute the camera intrinsics
        data[f"K_cam{i}"] = p_rect[i][0:3, 0:3]

    # Create 4x4 matrices from the rectifying rotation matrices
    r_rect = None
    if "R_rect_00" in filedata:
        r_rect = [np.eye(4) for _ in range(4)]
        for i in range(4):
            r_rect[i][0:3, 0:3] = np.reshape(filedata["R_rect_0" + str(i)], (3, 3))
            data[f"R_rect_{i}0"] = r_rect[i]

    # Load the rigid transformation from velodyne coordinates
    # to unrectified cam0 coordinates
    T_cam0unrect_velo = None
    stereo_baseline = None
    if velo_to_cam_file is not None and r_rect is not None:
        T_cam0unrect_velo = load_calib_rigid(velo_to_cam_file)

        velo_to_cam = [rectified_extrinsics[i].dot(r_rect[i].dot(T_cam0unrect_velo)) for i in range(4)]
        p_cam = np.array([0, 0, 0, 1])
        stereo_baseline = [np.linalg.inv(velo_to_cam[i]).dot(p_cam) for i in range(4)]

        for i in range(4):
            data[f"T_cam{i}_velo"] = velo_to_cam[i]

    elif "Tr_velo_to_cam" in filedata or "Tr_velo_cam" in filedata or "Tr" in filedata:
        prop_name = (
            "Tr_velo_to_cam" if "Tr_velo_to_cam" in filedata else "Tr_velo_cam" if "Tr_velo_cam" in filedata else "Tr"
        )
        data["T_cam0_velo"] = np.reshape(filedata[prop_name], (3, 4))
        data["T_cam0_velo"] = np.vstack([data["T_cam0_velo"], [0, 0, 0, 1]])
        for i in range(1, 4):
            data[f"T_cam{i}_velo"] = rectified_extrinsics[i].dot(data["T_cam0_velo"])
        p_cam = np.array([0, 0, 0, 1])
        stereo_baseline = [np.linalg.inv(data[f"T_cam{i}_velo"]).dot(p_cam) for i in range(4)]

    if stereo_baseline is not None:
        data["b_gray"] = np.linalg.norm(stereo_baseline[1] - stereo_baseline[0])
        data["b_rgb"] = np.linalg.norm(stereo_baseline[3] - stereo_baseline[2])

    return data
