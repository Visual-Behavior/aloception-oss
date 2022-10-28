import os
import numpy as np

# Disable Tensorflow loggings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # only TF error messages are printed
try:
    import tensorflow as tf
    import waymo_open_dataset
    import tqdm
except:
    raise Exception("Please install the requirements in alodataset/prepare/waymo-requirements.txt")


from multiprocessing import Pool
from os.path import join, isdir, basename
import argparse
from glob import glob
import pickle

from waymo_open_dataset.utils.frame_utils import parse_range_image_and_camera_projection
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import frame_utils

from scipy.ndimage import binary_closing, grey_opening
from scipy.interpolate import LinearNDInterpolator

# Abbreviations:
# WOD: Waymo Open Dataset
# FOV: field of view
# SDC: self-driving car
# 3dbox: 3D bounding box

# Some 3D bounding boxes do not contain any points
# This switch, when set True, filters these boxes
# It is safe to filter these boxes because they are not counted towards evaluation anyway
filter_empty_3dboxes = False


# There is no bounding box annotations in the No Label Zone (NLZ)
# if set True, points in the NLZ are filtered
filter_no_label_zone_points = True


# Only bounding boxes of certain classes are converted
# Note: Waymo Open Dataset evaluates for ALL_NS, including only 'VEHICLE', 'PEDESTRIAN', 'CYCLIST'
selected_waymo_classes = [
    # 'UNKNOWN',
    "VEHICLE",
    "PEDESTRIAN",
    # 'SIGN',
    "CYCLIST",
]


# Only data collected in specific locations will be converted
# If set None, this filter is disabled (all data will thus be converted)
# Available options: location_sf (main dataset)
selected_waymo_locations = None

# Save track id
save_track_id = True


class Waymo2KITTIConverter(object):
    def __init__(self, load_dir, save_dir, num_proc=1):
        """
        Parameters
        ----------
        load_dir : str
            directory path containing Waymo Open Dataset tfrecord files
        save_dir : str
            directory path to save parsed images and pickle files
        num_proc : int, optional
            number of processes, by default 1
        """
        # turn on eager execution for older tensorflow versions
        if int(tf.__version__.split(".")[0]) < 2:
            tf.enable_eager_execution()

        self.lidar_list = ["_FRONT", "_FRONT_RIGHT", "_FRONT_LEFT", "_SIDE_RIGHT", "_SIDE_LEFT"]
        self.type_list = ["UNKNOWN", "VEHICLE", "PEDESTRIAN", "SIGN", "CYCLIST"]
        self.waymo_to_kitti_class_map = {
            "UNKNOWN": "DontCare",
            "PEDESTRIAN": "Pedestrian",
            "VEHICLE": "Car",
            "CYCLIST": "Cyclist",
            "SIGN": "Sign",  # not in kitti
        }

        self.load_dir = load_dir
        self.save_dir = save_dir
        self.num_proc = int(num_proc)

        self.tfrecord_pathnames = sorted(glob(join(self.load_dir, "*.tfrecord")))

        self.label_save_dir = "label"
        self.label_all_save_dir = "label_all"
        self.image_save_dir = "image"
        self.depth_save_dir = "depth"
        self.calib_save_dir = "calib"
        self.point_cloud_save_dir = "velodyne"
        self.pose_save_dir = "pose"

    def convert(self):
        """Do the conversion.
        If the segment directory exists, check it's integrity.
        If OK, do nothing. If not, do the conversion.
        """
        print("start converting ...")
        with Pool(self.num_proc) as p:
            r = list(tqdm.tqdm(p.imap(self.convert_one, range(len(self))), total=len(self)))

    def convert_one(self, file_idx):
        pathname = self.tfrecord_pathnames[file_idx]
        self.convert_file(pathname)

    def check_segment_dir_integrity(self, sgmt_dir):
        # check pickle files
        calib_file = join(sgmt_dir, self.calib_save_dir + ".pkl")
        pose_file = join(sgmt_dir, self.pose_save_dir + ".pkl")
        lidar_label_file = join(sgmt_dir, "lidar_" + self.label_save_dir + ".pkl")
        cam_label_file = join(sgmt_dir, "camera_" + self.label_save_dir + ".pkl")
        for pkl_file in [calib_file, pose_file, lidar_label_file, cam_label_file]:
            if not os.path.exists(pkl_file):
                return False

        # check image & depth
        img_nb = []
        depth_nb = []
        for i in range(5):
            img_dir = join(sgmt_dir, self.image_save_dir + str(i))
            if not isdir(img_dir):
                return False
            else:
                img_nb.append(len([img for img in os.listdir(img_dir) if img.endswith("jpg")]))

            depth_dir = join(sgmt_dir, self.depth_save_dir + str(i))
            if not isdir(depth_dir):
                return False
            else:
                depth_nb.append(len([dpt for dpt in os.listdir(depth_dir) if dpt.endswith("npz")]))

        # image and depth numbers in each subdirs should be equal
        if sum(img_nb) / len(img_nb) != img_nb[0] or sum(depth_nb) / len(depth_nb) != img_nb[0]:
            return False

        elif len([dpt for dpt in os.listdir(depth_dir) if dpt.endswith("npz")]) != img_nb[0]:
            return False

        # if all the checks are passed
        return True

    def convert_file(self, pathname):
        dataset = tf.data.TFRecordDataset(pathname, compression_type="")
        sgmt_name = basename(pathname).split("_with_camera_labels")[0]
        sgmt_dir = join(self.save_dir, sgmt_name)

        # if segment directory exists and fully converted, do nothing
        if isdir(sgmt_dir) and self.check_segment_dir_integrity(sgmt_dir):
            return None

        self.create_folder(sgmt_name)

        calibs = {}
        lidar_labels = {}
        camera_labels = {}
        poses = {}

        for frame_idx, data in enumerate(dataset):
            # print(frame_idx)
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            if selected_waymo_locations is not None and frame.context.stats.location not in selected_waymo_locations:
                continue

            # save images
            self.save_image(frame, frame_idx, sgmt_name)

            self.save_dense_depth(frame, frame_idx, sgmt_name)

            # parse calibration files
            calibs[frame_idx] = self.save_calib(frame, frame_idx, sgmt_name)

            # parse point clouds
            # self.save_lidar(frame, frame_idx, sgmt_name)

            # parse label files
            lidar_labels[frame_idx] = self.save_lidar_label(frame, frame_idx, sgmt_name)

            camera_labels[frame_idx] = self.save_camera_label(frame, frame_idx, sgmt_name)

            # parse pose files
            poses[frame_idx] = self.save_pose(frame, frame_idx, sgmt_name)

        calib_file = join(self.save_dir, sgmt_name, self.calib_save_dir + ".pkl")
        with open(calib_file, "wb") as f:
            pickle.dump(calibs, f, pickle.HIGHEST_PROTOCOL)

        pose_file = join(self.save_dir, sgmt_name, self.pose_save_dir + ".pkl")
        with open(pose_file, "wb") as f:
            pickle.dump(poses, f, pickle.HIGHEST_PROTOCOL)

        label_file = join(self.save_dir, sgmt_name, "lidar_" + self.label_save_dir + ".pkl")
        with open(label_file, "wb") as f:
            pickle.dump(lidar_labels, f, pickle.HIGHEST_PROTOCOL)

        label_file = join(self.save_dir, sgmt_name, "camera_" + self.label_save_dir + ".pkl")
        with open(label_file, "wb") as f:
            pickle.dump(camera_labels, f, pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.tfrecord_pathnames)

    def save_image(self, frame, frame_idx, sgmt_name):
        """Parse and save images in jpg format

        Parameters
        ----------
        frame : waymo_open_dataset.dataset_pb2.Frame
            waymo open dataset frame proto
        frame_idx : int
            the current frame number
        sgmt_name : str
            the current segment name
        """
        for img in frame.images:
            img_path = join(
                self.save_dir, sgmt_name, self.image_save_dir + str(img.name - 1), str(frame_idx).zfill(3) + ".jpg"
            )
            with open(img_path, "wb") as f:
                f.write(img.image)

    def save_dense_depth(self, frame, frame_idx, sgmt_name):
        """Use lidar data to generate dense depth, pixels with no values are at 0

        Parameters
        ----------
        frame : waymo_open_dataset.dataset_pb2.Frame
            waymo open dataset frame proto
        frame_idx : int
            the current frame number
        sgmt_name : str
            the current segment name
        """

        (range_images, camera_projections, range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(
            frame
        )
        points, cp_points = frame_utils.convert_range_image_to_point_cloud(
            frame, range_images, camera_projections, range_image_top_pose
        )

        # 3d points in vehicle frame.
        points_all = np.concatenate(points, axis=0)

        # camera projection corresponding to each point.
        cp_points = np.concatenate(cp_points, axis=0)
        images = sorted(frame.images, key=lambda i: i.name)
        points_all = tf.norm(points_all, axis=-1, keepdims=True)
        cp_points = tf.constant(cp_points, dtype=tf.int32)

        for img in images:

            # lidar points can be on 2 cameras so we have to filter out other cameras
            mask0 = cp_points[..., 0] == img.name
            mask3 = cp_points[..., 3] == img.name
            # in some rare cases a single point can have multiple pair of coordinates for the same camera,
            # remove 2nd pair
            mask3 = mask3 & ~mask0
            mask = tf.concat([tf.tile(mask0[..., None], (1, 3)), tf.tile(mask3[..., None], (1, 3))], -1)
            cp_points_tensor = tf.cast(tf.reshape(cp_points[mask], (-1, 3)), dtype=tf.float32)
            points_tensor = points_all[mask0 | mask3]

            # project points to image scaled down by 4
            projected_points_all_from_raw_data = tf.concat(
                [cp_points_tensor[:, 1:] // 4, points_tensor], axis=-1
            ).numpy()
            s = tuple(tf.image.decode_jpeg(img.image).shape)
            image = np.zeros((round(s[0] / 4), round(s[1] / 4), 1))
            x = projected_points_all_from_raw_data[:, 0].astype(int)
            y = projected_points_all_from_raw_data[:, 1].astype(int)
            image[y, x] = projected_points_all_from_raw_data[:, 2, None]

            # select points that have lidar info
            data = image.squeeze()
            mask = np.where((data != 0))

            # select points to fill with interpolation
            structure = np.array(
                [
                    [0, 0, 1, 1, 1, 1, 1, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 1, 1, 1, 1, 0],
                    [0, 0, 1, 1, 1, 1, 1, 0, 0],
                ]
            )
            m = binary_closing((image > 0.0).squeeze(), structure=structure)
            to_fill = np.where(m)

            # linear interpolation between points
            try:
                interp = LinearNDInterpolator(mask, data[mask], fill_value=0.0)
                res = np.full(image.shape[:-1], np.nan)
                filled_data = interp(*to_fill)
                res[to_fill[0], to_fill[1]] = filled_data

                # filter results to remove artefacts by favoring reducing the range
                footprint = np.array(
                    [
                        [0, 0, 1, 1, 1, 0, 0],
                        [0, 1, 1, 1, 1, 1, 0],
                        [0, 1, 1, 1, 1, 1, 0],
                        [0, 1, 1, 1, 1, 1, 0],
                        [0, 1, 1, 1, 1, 1, 0],
                        [0, 1, 1, 1, 1, 1, 0],
                        [0, 0, 1, 1, 1, 0, 0],
                    ]
                )
                res = grey_opening(res, footprint=footprint)
                res = np.nan_to_num(res)
            except:
                res = np.zeros(image.shape[:-1])
            img_path = join(self.save_dir, sgmt_name, self.depth_save_dir + str(img.name - 1), str(frame_idx).zfill(3))
            np.savez_compressed(img_path, (res * 100).astype(np.ushort))

    def save_calib(self, frame, frame_idx, sgmt_name):
        """Parse and return camera calibration as dict of np.ndarray

        Parameters
        ----------
        frame : waymo_open_dataset.dataset_pb2.Frame
            waymo open dataset frame proto
        frame_idx : int
            the current frame number
        sgmt_name : str
            the current segment name

        Returns
        -------
        dict of np.ndarray
            keys: "cam_intrinsic", "cam_extrinsic"
        """
        raw_context = {}
        raw_context["cam_intrinsic"] = {}
        raw_context["cam_extrinsic"] = {}

        for camera in frame.context.camera_calibrations:
            cam_to_vehicle = np.array(camera.extrinsic.transform).reshape(4, 4)

            cam_intrinsic = np.zeros((3, 4))
            cam_intrinsic[0, 0] = camera.intrinsic[0]
            cam_intrinsic[1, 1] = camera.intrinsic[1]
            cam_intrinsic[0, 2] = camera.intrinsic[2]
            cam_intrinsic[1, 2] = camera.intrinsic[3]
            cam_intrinsic[2, 2] = 1

            raw_context["cam_intrinsic"][camera.name] = cam_intrinsic
            raw_context["cam_extrinsic"][camera.name] = cam_to_vehicle
        return raw_context

    def save_camera_label(self, frame, frame_idx, sgmt_name):
        """Parse and return labels 2d

        Parameters
        ----------
        frame : waymo_open_dataset.dataset_pb2.Frame
            waymo open dataset frame proto
        frame_idx : int
            the current frame number
        sgmt_name : str
            the current segment name

        Returns
        -------
        dict
            keys: "0", "1", "2", "3", "4" corresponding to each camera
            values: list of dict containing labels 2d "bbox", "camera_id", "track_id", "class"
        """
        labels = {
            "0": [],
            "1": [],
            "2": [],
            "3": [],
            "4": [],
        }
        for camera in frame.camera_labels:
            name = str(camera.name - 1)
            for obj in camera.labels:
                bbox = [obj.box.center_x, obj.box.center_y, obj.box.length, obj.box.width]

                item = {"bbox": bbox, "camera_id": name, "track_id": obj.id, "class": obj.type}

                labels[str(name)].append(item)
        return labels

    def save_lidar_label(self, frame, frame_idx, sgmt_name):
        """Parse and return labels 3d

        Parameters
        ----------
        frame : waymo_open_dataset.dataset_pb2.Frame
            waymo open dataset frame proto
        frame_idx : int
            the current frame number
        sgmt_name : str
            the current segment name

        Returns
        -------
        dict
            keys: "0", "1", "2", "3", "4" corresponding to each camera
            values: list of dict containing labels 3d "bbox_proj",  "bbox_3d", "camera_id", "track_id", "class", "speed", "accel"
        """
        # lbl_path = join(self.save_dir, sgmt_name, self.label_all_save_dir, str(frame_idx).zfill(3) + '.bin')

        # preprocess bounding box data
        id_to_bbox = dict()
        id_to_name = dict()

        for labels in frame.projected_lidar_labels:
            name = labels.name
            for label in labels.labels:
                # waymo: bounding box origin is at the center
                # TODO: need a workaround as bbox may not belong to front cam
                bbox = [label.box.center_x, label.box.center_y, label.box.length, label.box.width]
                id_to_bbox[label.id] = bbox
                id_to_name[label.id] = name - 1

        labels = {
            "0": [],
            "1": [],
            "2": [],
            "3": [],
            "4": [],
        }

        # print([i.type for i in frame.laser_labels])
        for obj in frame.laser_labels:
            # calculate bounding box
            bounding_box = None
            name = None
            id = obj.id
            for lidar in self.lidar_list:
                if id + lidar in id_to_bbox:
                    bounding_box = id_to_bbox.get(id + lidar)
                    name = str(id_to_name.get(id + lidar))
                    break

            # TODO: temp fix
            if bounding_box == None or name == None:
                # name = '0'
                # bounding_box = (0, 0, 0, 0)
                continue

            my_type = self.type_list[obj.type]

            if my_type not in selected_waymo_classes:
                continue

            if filter_empty_3dboxes and obj.num_lidar_points_in_box < 1:
                continue

            # track id
            track_id = obj.id

            box_3d = [
                obj.box.center_x,
                obj.box.center_y,
                obj.box.center_z,
                obj.box.length,
                obj.box.width,
                obj.box.height,
                obj.box.heading,
            ]

            item = {
                "bbox_proj": bounding_box,
                "bbox_3d": box_3d,
                "camera_id": name,
                "track_id": track_id,
                "class": obj.type,
                "speed": [obj.metadata.speed_x, obj.metadata.speed_y],
                "accel": [obj.metadata.accel_x, obj.metadata.accel_y],
            }

            labels[name].append(item)

        return labels

    def save_pose(self, frame, frame_idx, sgmt_name):
        """Save self driving car (SDC)'s own pose

        Parameters
        ----------
        frame : waymo_open_dataset.dataset_pb2.Frame
            waymo open dataset frame proto
        frame_idx : int
            the current frame number
        sgmt_name : str
            the current segment name

        Returns
        -------
        np.ndarray

        Note
        -----
        Note that SDC's own pose is not included in the regular training of KITTI dataset
        KITTI raw dataset contains ego motion files but are not often used
        Pose is important for algorithms that takes advantage of the temporal information

        """
        pose = np.array(frame.pose.transform).reshape(4, 4)
        return pose

    def create_folder(self, sgmt_name):
        d = join(self.save_dir, sgmt_name, self.point_cloud_save_dir)
        if not isdir(d):
            os.makedirs(d)

        for i in range(5):
            d = join(self.save_dir, sgmt_name, self.image_save_dir + str(i))
            if not isdir(d):
                os.makedirs(d)

            d = join(self.save_dir, sgmt_name, self.depth_save_dir + str(i))
            if not isdir(d):
                os.makedirs(d)

    def cart_to_homo(self, mat):
        ret = np.eye(4)
        if mat.shape == (3, 3):
            ret[:3, :3] = mat
        elif mat.shape == (3, 4):
            ret[:3, :] = mat
        else:
            raise ValueError(mat.shape)
        return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("load_dir", help="Directory to load Waymo Open Dataset tfrecords")
    parser.add_argument("save_dir", help="Directory to save converted KITTI-format data")
    parser.add_argument("--num_proc", default=1, help="Number of processes to spawn")
    args = parser.parse_args()

    converter = Waymo2KITTIConverter(args.load_dir, args.save_dir, args.num_proc)
    converter.convert()
