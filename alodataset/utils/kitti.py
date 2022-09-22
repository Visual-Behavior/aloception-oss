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
    """P_rect_00 = np.reshape(filedata["P_rect_00"], (3, 4))
    P_rect_10 = np.reshape(filedata["P_rect_01"], (3, 4))
    P_rect_20 = np.reshape(filedata["P_rect_02"], (3, 4))
    P_rect_30 = np.reshape(filedata["P_rect_03"], (3, 4))"""
    p_rect = [np.reshape(filedata[p], (3, 4)) for p in names]

    """data["P_rect_00"] = P_rect_00
    data["P_rect_10"] = P_rect_10
    data["P_rect_20"] = P_rect_20
    data["P_rect_30"] = P_rect_30"""
    for i, p in enumerate(p_rect):
        data[f"P_rect_{i}0"] = p

    # Compute the rectified extrinsics from cam0 to camN
    """T0 = np.eye(4)
    T0[0, 3] = P_rect_00[0, 3] / P_rect_00[0, 0]
    T1 = np.eye(4)
    T1[0, 3] = P_rect_10[0, 3] / P_rect_10[0, 0]
    T2 = np.eye(4)
    T2[0, 3] = P_rect_20[0, 3] / P_rect_20[0, 0]
    T3 = np.eye(4)
    T3[0, 3] = P_rect_30[0, 3] / P_rect_30[0, 0]"""

    rectified_extrinsics = [np.eye(4) for _ in range(4)]
    for i in range(4):
        rectified_extrinsics[i][0, 3] = p_rect[i][0, 3] / p_rect[i][0, 0]
        data[f"T_cam{i}_rect"] = rectified_extrinsics[i]
        data[f"K_cam{i}"] = p_rect[i][0:3, 0:3]

    # Create 4x4 matrices from the rectifying rotation matrices
    """R_rect_00 = np.eye(4)
    R_rect_00[0:3, 0:3] = np.reshape(filedata["R_rect_00"], (3, 3))
    R_rect_10 = np.eye(4)
    R_rect_10[0:3, 0:3] = np.reshape(filedata["R_rect_01"], (3, 3))
    R_rect_20 = np.eye(4)
    R_rect_20[0:3, 0:3] = np.reshape(filedata["R_rect_02"], (3, 3))
    R_rect_30 = np.eye(4)
    R_rect_30[0:3, 0:3] = np.reshape(filedata["R_rect_03"], (3, 3))"""

    r_rect = None
    if "R_rect_00" in filedata:
        r_rect = [np.eye(4) for _ in range(4)]
        for i in range(4):
            r_rect[i][0:3, 0:3] = np.reshape(filedata["R_rect_0" + str(i)], (3, 3))
            data[f"R_rect_{i}0"] = r_rect[i]

    """data["R_rect_00"] = R_rect_00
    data["R_rect_10"] = R_rect_10
    data["R_rect_20"] = R_rect_20
    data["R_rect_30"] = R_rect_30

    data["T_cam0_rect"] = T0
    data["T_cam1_rect"] = T1
    data["T_cam2_rect"] = T2
    data["T_cam3_rect"] = T3"""

    # Load the rigid transformation from velodyne coordinates
    # to unrectified cam0 coordinates
    T_cam0unrect_velo = None
    if velo_to_cam_file is not None and r_rect is not None:
        T_cam0unrect_velo = load_calib_rigid(velo_to_cam_file)

        velo_to_cam = [rectified_extrinsics[i].dot(r_rect[i].dot(T_cam0unrect_velo)) for i in range(4)]
        p_cam = np.array([0, 0, 0, 1])
        stereo_baseline = [np.linalg.inv(velo_to_cam[i]).dot(p_cam) for i in range(4)]
        data["b_gray"] = np.linalg.norm(stereo_baseline[1] - stereo_baseline[0])
        data["b_rgb"] = np.linalg.norm(stereo_baseline[3] - stereo_baseline[2])

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
        data["b_gray"] = np.linalg.norm(stereo_baseline[1] - stereo_baseline[0])
        data["b_rgb"] = np.linalg.norm(stereo_baseline[3] - stereo_baseline[2])

    """data["T_cam0_velo"] = T0.dot(R_rect_00.dot(T_cam0unrect_velo))
    data["T_cam1_velo"] = T1.dot(R_rect_00.dot(T_cam0unrect_velo))
    data["T_cam2_velo"] = T2.dot(R_rect_00.dot(T_cam0unrect_velo))
    data["T_cam3_velo"] = T3.dot(R_rect_00.dot(T_cam0unrect_velo))"""

    # Compute the camera intrinsics
    """data["K_cam0"] = P_rect_00[0:3, 0:3]
    data["K_cam1"] = P_rect_10[0:3, 0:3]
    data["K_cam2"] = P_rect_20[0:3, 0:3]
    data["K_cam3"] = P_rect_30[0:3, 0:3]"""

    # Compute the stereo baselines in meters by projecting the origin of
    # each camera frame into the velodyne frame and computing the distances
    # between them
    """p_cam = np.array([0, 0, 0, 1])
    p_velo0 = np.linalg.inv(data["T_cam0_velo"]).dot(p_cam)
    p_velo1 = np.linalg.inv(data["T_cam1_velo"]).dot(p_cam)
    p_velo2 = np.linalg.inv(data["T_cam2_velo"]).dot(p_cam)
    p_velo3 = np.linalg.inv(data["T_cam3_velo"]).dot(p_cam)

    data["b_gray"] = np.linalg.norm(p_velo1 - p_velo0)  # gray baseline
    data["b_rgb"] = np.linalg.norm(p_velo3 - p_velo2)  # rgb baseline"""

    return data
