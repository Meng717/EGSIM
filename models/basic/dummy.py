from models.basic.basic_class import BasicModelLoader, BasicSensorInfo, \
                                     BasicSensorLoader, BasicRenderer
from models.basic.utils.general_utils import pad_poses
from dataclasses import dataclass
from typing import Literal
import numpy as np
import glob
import os

@dataclass
class DummyRadarInfo(BasicSensorInfo):
    pass

@dataclass
class DummyLidarInfo(BasicSensorInfo):
    pass

@dataclass
class DummyPcdInfo:
    timestamp: int
    sensor: Literal['lidar', 'radar']
    points: np.ndarray

class DummyPcdLoader(BasicModelLoader):

    def set(self, data_path, model_path, load_lidar=True):
        super().set(data_path, model_path)
        self.load_lidar = load_lidar
        if self.load_lidar:
            self.point_dir_path_list = sorted(glob.glob(
                os.path.join(self.model_path, "lidar_*")))
        else:
            self.point_dir_path_list = sorted(glob.glob(
                os.path.join(self.model_path, "radar_*")))

    def load(self) -> dict[DummyPcdInfo]:
        pcd_info_dict = {}
        frame_num = len(glob.glob(os.path.join(self.data_path, "pose", "*.txt")))
        
        for idx, point_dir in enumerate(self.point_dir_path_list):
            point_list = sorted(os.listdir(point_dir))
            for i in range(frame_num):
                input_bin = os.path.join(point_dir, point_list[i])
                if self.load_lidar:
                    pcd = np.fromfile(input_bin, dtype=np.float32, count=-1).reshape([-1, 6])[:, :3]
                else:
                    pcd = np.fromfile(input_bin, dtype=np.float32, count=-1).reshape([-1, 2])
                    pcd = np.concatenate([pcd, np.zeros((pcd.shape[0], 1))], axis=1)
                pcd_info_dict.setdefault(idx, []).append(
                    DummyPcdInfo(timestamp=i, sensor='lidar' if self.load_lidar else 'radar', points=pcd))
        return pcd_info_dict

class DummyLidarLoader(BasicSensorLoader):

    def load(self) -> dict[DummyLidarInfo]:
        cam_infos = {}

        for ego_pose in self.ego_poses:
            R = ego_pose[:3, :3]
            T = ego_pose[:3, 3]
            cam_infos.setdefault(0, []).append(
                DummyLidarInfo(idx=0, rotation=R, position=T)
            )
        return cam_infos

class DummyRadarLoader(BasicSensorLoader):

    def load(self) -> dict[DummyRadarInfo]:
        cam_num = len(glob.glob(os.path.join(
            self.model_path, "radar", "radar_*")))
        cam_infos = {}
        calib_list = sorted(glob.glob(os.path.join(self.data_path, 'calib', '*.txt')))
        
        radar2lidars = []
        for calib_file in calib_list:
            with open(calib_file) as f:
                lines = f.readlines()
            target_line_index = None
            for i, line in enumerate(lines):
                if "T_lidar2radar0:" in line:
                    target_line_index = i
                    break
            data_lines = lines[target_line_index : target_line_index + cam_num]
            lidar2radar = []
            for line in data_lines:
                data_part = line.split(':', 1)[1].strip()
                numbers = list(map(float, data_part.split()))
                matrix = np.array(numbers).reshape(3, 4)
                lidar2radar.append(matrix)
            lidar2radar = pad_poses(np.array(lidar2radar))
            radar2lidar = np.linalg.inv(lidar2radar)
            radar2lidars.append(radar2lidar)

        r2w = np.array(self.ego_poses)[:, None, ...] @ np.array(radar2lidars)
        for frame_idx in range(self.frame_num):
            for cam_idx in range(cam_num):
                R = r2w[frame_idx, cam_idx, :3, :3]
                T = r2w[frame_idx, cam_idx, :3, 3]
                cam_infos.setdefault(cam_idx, []).append(
                    DummyRadarInfo(idx=cam_idx, rotation=R, position=T)
                )
        return cam_infos

class DummyLidarRenderer(BasicRenderer):
    def set(self):
        pass
    
    def getSettings(self) -> None:
        pass
    
    def razterization(self, lidar: DummyLidarInfo, model_info: DummyPcdInfo) -> np.ndarray:
        now_lidar2world = lidar.c2w.cpu().numpy()
        init_lidar2world = lidar.init_c2w.cpu().numpy()
        init2now = np.linalg.inv(now_lidar2world) @ init_lidar2world
        points = (np.concatenate([model_info.points, np.ones((model_info.points.shape[0], 1))], axis=1) \
            @ init2now.T)[:, :3]
        return points

class DummyRadarRenderer(BasicRenderer):
    def set(self):
        pass
    
    def getSettings(self) -> None:
        pass
    
    def razterization(self, radar: DummyRadarInfo, model_info: DummyPcdInfo) -> np.ndarray:
        now_radar2world = radar.c2w.cpu().numpy()
        init_radar2world = radar.init_c2w.cpu().numpy()
        init2now = np.linalg.inv(now_radar2world) @ init_radar2world
        points = (np.concatenate([model_info.points, np.ones((model_info.points.shape[0], 1))], axis=1) \
            @ init2now.T)[:, :3]
        return points