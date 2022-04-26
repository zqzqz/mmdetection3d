import mmcv
import numpy as np
from glob import glob
import os
import shutil
import pickle
import open3d as o3d
from collections import OrderedDict
from pathlib import Path
from concurrent import futures as futures
from skimage import io

from .kitti_data_utils import get_label_anno
from mmdet3d.core.bbox import box_np_ops


def _calculate_num_points_in_gt(data_path,
                                infos,
                                relative_path,
                                remove_outside=False,
                                num_features=4):
    for info in mmcv.track_iter_progress(infos):
        pc_info = info['point_cloud']
        if relative_path:
            v_path = str(Path(data_path) / pc_info['velodyne_path'])
        else:
            v_path = pc_info['velodyne_path']
        points_v = np.fromfile(
            v_path, dtype=np.float32, count=-1).reshape([-1, num_features])

        # points_v = points_v[points_v[:, 0] > 0]
        annos = info['annos']
        num_obj = len([n for n in annos['name'] if n != 'DontCare'])
        # annos = kitti.filter_kitti_anno(annos, ['DontCare'])
        dims = annos['dimensions'][:num_obj]
        loc = annos['location'][:num_obj]
        rots = annos['rotation_y'][:num_obj]
        gt_boxes_lidar = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                         axis=1)
        indices = box_np_ops.points_in_rbbox(points_v[:, :3], gt_boxes_lidar)
        num_points_in_gt = indices.sum(0)
        num_ignored = len(annos['dimensions']) - num_obj
        num_points_in_gt = np.concatenate(
            [num_points_in_gt, -np.ones([num_ignored])])
        annos['num_points_in_gt'] = num_points_in_gt.astype(np.int32)


def get_opv2v_image_info(path,
                         mode="train",
                         image_ids=7481,
                         num_worker=8,
                         relative_path=True):
    """
    opv2v annotation format version 2:
    {
        [optional]points: [N, 3+] point cloud
        [optional, for opv2v]image: {
            image_idx: ...
            image_path: ...
            image_shape: ...
        }
        point_cloud: {
            num_features: 4
            velodyne_path: ...
        }
        [optional, for opv2v]calib: {
            R0_rect: ...
            Tr_velo_to_cam: ...
            P2: ...
        }
        annos: {
            location: [num_gt, 3] array
            dimensions: [num_gt, 3] array
            rotation_y: [num_gt] angle array
            name: [num_gt] ground truth name array
            [optional]difficulty: opv2v difficulty
            [optional]group_ids: used for multi-part object
        }
    }
    """
    root_path = path
    if not isinstance(image_ids, list):
        image_ids = list(range(image_ids))

    def map_func(idx):
        info = {}
        point_cloud_path = os.path.join(root_path if not relative_path else ".", mode, "velodyne", "{:06d}.bin".format(idx))
        info["point_cloud"] = {
            "num_features": 4,
            "velodyne_path": point_cloud_path
        }
        label_path = os.path.join(root_path if not relative_path else ".", mode, "label_2", "{:06d}.txt".format(idx))
        info["annos"] = get_label_anno(os.path.join(root_path, label_path) if relative_path else label_path)
        # do not use camera format of dimension
        info['annos']['dimensions'] = info['annos']['dimensions'][:,[1, 2, 0]]
        return info

    with futures.ThreadPoolExecutor(num_worker) as executor:
        image_infos = executor.map(map_func, image_ids)

    return list(image_infos)


def create_opv2v_info_file(data_path,
                           pkl_prefix='opv2v',
                           save_path=None,
                           relative_path=True):
    """Create info file of OPV2V dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        data_path (str): Path of the data root.
        pkl_prefix (str): Prefix of the info file to be generated.
        save_path (str): Path to save the info file.
        relative_path (bool): Whether to use relative path.
    """
    if save_path is None:
        save_path = Path(data_path)
    else:
        save_path = Path(save_path)
    opv2v_infos_train = get_opv2v_image_info(
        data_path,
        mode="train",
        image_ids=7047,
        relative_path=relative_path)
    _calculate_num_points_in_gt(data_path, opv2v_infos_train, relative_path)
    filename = save_path / f'{pkl_prefix}_infos_train.pkl'
    print(f'opv2v info train file is saved to {filename}')
    mmcv.dump(opv2v_infos_train, filename)
    opv2v_infos_val = get_opv2v_image_info(
        data_path,
        mode="validate",
        image_ids=5239,
        relative_path=relative_path)
    _calculate_num_points_in_gt(data_path, opv2v_infos_val, relative_path)
    filename = save_path / f'{pkl_prefix}_infos_val.pkl'
    print(f'opv2v info val file is saved to {filename}')
    mmcv.dump(opv2v_infos_val, filename)
    filename = save_path / f'{pkl_prefix}_infos_trainval.pkl'
    print(f'opv2v info trainval file is saved to {filename}')
    mmcv.dump(opv2v_infos_train + opv2v_infos_val, filename)

    opv2v_infos_test = get_opv2v_image_info(
        data_path,
        mode="test",
        image_ids=5985,
        relative_path=relative_path)
    filename = save_path / f'{pkl_prefix}_infos_test.pkl'
    print(f'opv2v info test file is saved to {filename}')
    mmcv.dump(opv2v_infos_test, filename)


def read_pcd(filepath):
    pcd = o3d.io.read_point_cloud(filepath)
    pcd = np.asarray(pcd.points).astype(np.float32)
    if pcd.shape[1] == 3:
        pcd = np.hstack(
            (pcd, 
             np.ones((pcd.shape[0], 1)) * 0.5)
        ).astype(np.float32)
    return pcd

def write_bin(pcd_data: np.ndarray, filepath):
    pcd_data.tofile(filepath)

def pcd_to_bin(pcd_path, bin_path):
    pcd_data = read_pcd(pcd_path)
    write_bin(pcd_data, bin_path)

def rotation_matrix(pitch, yaw, roll):
    R = np.array([[np.cos(yaw)*np.cos(pitch), 
                    np.cos(yaw)*np.sin(pitch)*np.sin(roll)-np.sin(yaw)*np.cos(roll), 
                    np.cos(yaw)*np.sin(pitch)*np.cos(roll)+np.sin(yaw)*np.sin(roll)],
                    [np.sin(yaw)*np.cos(pitch), 
                    np.sin(yaw)*np.sin(pitch)*np.sin(roll)+np.cos(yaw)*np.cos(roll), 
                    np.sin(yaw)*np.sin(pitch)*np.cos(roll)-np.cos(yaw)*np.sin(roll)],
                    [-np.sin(pitch), 
                    np.cos(pitch)*np.sin(roll), 
                    np.cos(pitch)*np.cos(roll)]])
    return R

def bbox_map_to_sensor(bbox, sensor_calib):
    sensor_location = sensor_calib[:3]
    sensor_rotation = sensor_calib[3:] * np.pi / 180
    new_bbox = np.copy(bbox)
    if bbox.ndim == 1:
        new_bbox[:3] -= sensor_location
        new_bbox[:3] = np.dot(
                        np.linalg.inv(rotation_matrix(*(sensor_rotation))),
                        new_bbox[:3].T).T
        new_bbox[6] -= sensor_rotation[1]
    elif bbox.ndim == 2:
        new_bbox[:,:3] -= sensor_location
        new_bbox[:,:3] = np.dot(
                        np.linalg.inv(rotation_matrix(*(sensor_rotation))),
                        new_bbox[:,:3].T).T
        new_bbox[:,6] -= sensor_rotation[1]
    else:
        raise Exception("Wrong dimension of bbox")
    return new_bbox


class OPV2VDataset():
    def __init__(self, root_path, mode):
        self.root_path = root_path
        self.mode = mode

        with open(os.path.join(root_path, "{}.pkl".format(mode)), 'rb') as f:
            self.meta = pickle.load(f)

        self.cases = {
            "single_vehicle": [],
            "multi_vehicle": []
        }
        self._build_cases()

    def _build_cases(self):
        for scenario_id, scenario_data in self.meta.items():
            for frame_id, frame_data in scenario_data["data"].items():
                self.cases["multi_vehicle"].append({
                    "scenario_id": scenario_id, 
                    "frame_id": frame_id
                })
                for vehicle_id, vehicle_data in frame_data.items():
                    self.cases["single_vehicle"].append({
                        "scenario_id": scenario_id, 
                        "frame_id": frame_id,
                        "vehicle_id": vehicle_id
                    })
        if self.mode == "train":
            self.cases["single_vehicle"] = self.cases["single_vehicle"][::3]
        elif self.mode == "validate":
            self.cases["single_vehicle"] = self.cases["single_vehicle"][::2]

    def _get_lidar(self, scenario_id, frame_id, vehicle_id):
        return self.meta[scenario_id]["data"][frame_id][vehicle_id]["lidar"]

    def _get_camera(self, scenario_id, frame_id, vehicle_id, camera_id=0):
        return self.meta[scenario_id]["data"][frame_id][vehicle_id]["camera0"]

    def _get_calib(self, scenario_id, frame_id, vehicle_id):
        return self.meta[scenario_id]["data"][frame_id][vehicle_id]["calib"]

    def case_number(self, tag="single_vehicle"):
        return len(self.cases[tag])

    def get_case(self, idx, tag="single_vehicle"):
        case_meta = self.cases[tag][idx]
        return self.get_case_by_meta(case_meta, tag)
    
    def get_case_by_meta(self, case_meta, tag="single_vehicle"):
        if tag == "single_vehicle":
            return {
                "lidar": self._get_lidar(**case_meta),
                "camera": self._get_camera(**case_meta),
                "calib": self._get_calib(**case_meta)
            }
        else:
            raise NotImplementedError


class OPV2V2KITTI(object):
    """OPV2V to opv2v converter.

    This class serves as the converter to change the waymo raw data to opv2v
    format.

    Args:
        load_dir (str): Directory to load waymo raw data.
        save_dir (str): Directory to save data in opv2v format.
        prefix (str): Prefix of filename. In general, 0 for training, 1 for
            validation and 2 for testing.
        workers (str): Number of workers for the parallel process.
        test_mode (bool): Whether in the test_mode. Default: False.
    """

    def __init__(self,
                 load_dir,
                 save_dir,
                 mode,
                 workers=8):
        self.load_dir = load_dir
        self.save_dir = save_dir
        self.workers = int(workers)
        self.mode = mode

        self.data_loader = OPV2VDataset(self.load_dir, mode)

        self.label_save_dir = "{}/{}/label_2".format(self.save_dir, mode)
        self.image_save_dir = "{}/{}/image_2".format(self.save_dir, mode)
        self.calib_save_dir = "{}/{}/calib".format(self.save_dir, mode)
        self.point_cloud_save_dir = "{}/{}/velodyne".format(self.save_dir, mode)

        self.create_folder()

    def create_folder(self):
        os.makedirs(self.label_save_dir, exist_ok=True)
        os.makedirs(self.image_save_dir, exist_ok=True)
        os.makedirs(self.calib_save_dir, exist_ok=True)
        os.makedirs(self.point_cloud_save_dir, exist_ok=True)

    def convert(self):
        """Convert action."""
        print('Start converting ...')
        # mmcv.track_parallel_progress(self.convert_one, range(self.data_loader.case_number()),
        #                              self.workers)
        for i in range(self.data_loader.case_number(tag="single_vehicle")):
            self.convert_one(i)
            print("{}/{}".format(i, self.data_loader.case_number(tag="single_vehicle")))
        print('\nFinished ...')

    def convert_one(self, file_idx):
        """Convert action for single file.

        Args:
            file_idx (int): Index of the file to be converted.
        """
        # if os.path.exists(os.path.join(self.label_save_dir, "{:06d}.txt".format(file_idx))):
        #     return

        # case_data = self.data_loader.get_case(file_idx)
        case_meta = self.data_loader.cases["single_vehicle"][file_idx]
        calib = self.data_loader._get_calib(**case_meta)
        
        # # camera image
        # os.symlink(case_data["camera"], os.path.join(self.image_save_dir, "{:06d}.png".format(file_idx)))
        
        # # pointcloud
        # pcd_to_bin(case_data["lidar"], os.path.join(self.point_cloud_save_dir, "{:06d}.bin".format(file_idx)))

        # label
        opv2v_labels = []
        for vehicle_id, vehicle_data in calib["vehicles"].items():
            bbox = np.array([*vehicle_data["location"], *((np.array(vehicle_data["extent"])*2).tolist()), vehicle_data["angle"][1]*np.pi/180])
            bbox = bbox_map_to_sensor(bbox, np.array(calib["lidar_pose"]))
            if bbox[6] > np.pi:
                bbox[6] -= 2 * np.pi
            if bbox[6] < -np.pi:
                bbox[6] += 2 * np.pi
            opv2v_labels.append([
                "Car", # all cars
                0, # set truncated to none
                0, # set occlusion to unknown 
                0, 0, 0, 0, 0, # set camera image info to void
                bbox[3], bbox[4], bbox[5],
                bbox[0], bbox[1], bbox[2],
                bbox[6]
            ])
        label_str = ""
        for label in opv2v_labels:
            label_str += "{} {:.2f} {:d} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}\n".format(*label)
        with open(os.path.join(self.label_save_dir, "{:06d}.txt".format(file_idx)) , 'w') as f:
            f.write(label_str)

        # # calib
        # T_front_cam_to_ref = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0],
        #                         [1.0, 0.0, 0.0]])
        # camera_calibs = []
        # R0_rect = [f'{i:e}' for i in np.eye(3).flatten()]
        # Tr_velo_to_cams = []
        # calib_context = ''

        # for i in range(4):
        #     camera_data = case_data["calib"]["camera{}".format(i)]
        #     # extrinsic parameters
        #     T_cam_to_vehicle = np.array(camera_data["extrinsic"]).reshape(
        #         4, 4)
        #     T_vehicle_to_cam = np.linalg.inv(T_cam_to_vehicle)
        #     Tr_velo_to_cam = \
        #         self.cart_to_homo(T_front_cam_to_ref) @ T_vehicle_to_cam
        #     Tr_velo_to_cam = Tr_velo_to_cam[:3, :].reshape((12, ))
        #     Tr_velo_to_cams.append([f'{i:e}' for i in Tr_velo_to_cam])

        #     # intrinsic parameters
        #     camera_calib = np.hstack([np.array(camera_data["intrinsic"]), np.zeros((3,1))])
        #     camera_calib = list(camera_calib.reshape(12))
        #     camera_calib = [f'{i:e}' for i in camera_calib]
        #     camera_calibs.append(camera_calib)

        # # all camera ids are saved as id-1 in the result because
        # # camera 0 is unknown in the proto
        # for i in range(4):
        #     calib_context += 'P' + str(i) + ': ' + \
        #         ' '.join(camera_calibs[i]) + '\n'
        # calib_context += 'R0_rect' + ': ' + ' '.join(R0_rect) + '\n'
        # calib_context += 'Tr_velo_to_cam' + ': ' + \
        #     ' '.join(Tr_velo_to_cams[2]) + '\n'

        # with open(os.path.join(self.calib_save_dir, "{:06d}.txt".format(file_idx)) , 'w') as f:
        #     f.write(calib_context)

    # def cart_to_homo(self, mat):
    #     """Convert transformation matrix in Cartesian coordinates to
    #     homogeneous format.

    #     Args:
    #         mat (np.ndarray): Transformation matrix in Cartesian.
    #             The input matrix shape is 3x3 or 3x4.

    #     Returns:
    #         np.ndarray: Transformation matrix in homogeneous format.
    #             The matrix shape is 4x4.
    #     """
    #     ret = np.eye(4)
    #     if mat.shape == (3, 3):
    #         ret[:3, :3] = mat
    #     elif mat.shape == (3, 4):
    #         ret[:3, :] = mat
    #     else:
    #         raise ValueError(mat.shape)
    #     return ret
