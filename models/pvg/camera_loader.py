import os
import numpy as np
import torch
import torch.nn.functional as F
from models.basic.basic_class import BasicSensorLoader
import kornia
from models.basic.cameras import PinholeCamInfo
from models.basic.utils.general_utils import pad_poses
from dataclasses import dataclass
import glob
import json

frame_interval = 0.02
time_duration = [-0.5, 0.5]
fix_radius = 15.0
H = 900
W = 1600

@dataclass
class PVGCameraInfo(PinholeCamInfo):
    timestamp: float = ...
    
    @property
    def grid(self):
        return kornia.utils.create_meshgrid(self.height, self.width, normalized_coordinates=False, 
                                            device=self.device, dtype=self.dtype)[0]
    @property
    def get_world_directions(self):
        u, v = self.grid.unbind(-1)
        directions = torch.stack([(u-self.cx+0.5)/self.fx,
                                    (v-self.cy+0.5)/self.fy,
                                    torch.ones_like(u)], dim=0)
        directions = F.normalize(directions, dim=0)
        directions = (self.transformed_c2w[:3, :3] @ directions.reshape(3, -1)).reshape(3, self.height, self.width)
        return directions.to(device=self.device, dtype=self.dtype)

class PVGCameraLoader(BasicSensorLoader):

    def load(self) -> dict[PVGCameraInfo]:
        global time_duration
        cam_infos = {}
        calib_list = sorted(glob.glob(os.path.join(self.data_path, 'calib', '*.txt')))
        if os.path.exists(os.path.join(self.model_path, 'cameras.json')):
            with open(os.path.join(self.model_path, 'cameras.json'), 'r', encoding='utf-8') as file:
                cam_data = json.load(file)
            cam_num = len(cam_data) // self.frame_num
            if frame_interval > 0:
                time_duration = [-frame_interval*(self.frame_num-1)/2,frame_interval*(self.frame_num-1)/2]
            else:
                time_duration = time_duration
            
            cam2lidars = []
            Ks_list = []
            for calib_file in calib_list:
                with open(calib_file) as f:
                    lines = f.readlines()
                p_target_line_index = None
                t_target_line_index = None
                for i, line in enumerate(lines):
                    if "P0:" in line:
                        p_target_line_index = i
                    if "T_lidar2cam0" in line:
                        t_target_line_index = i
                p_data_lines = lines[p_target_line_index : p_target_line_index + cam_num]
                t_data_lines = lines[t_target_line_index : t_target_line_index + cam_num]
                Ks = np.array([list(map(float, line.split()[1:])) for line in p_data_lines]).reshape(-1, 3, 4)[:, :, :3]
                Ks_list.append(Ks)
                lidar2cam = np.array([list(map(float, line.split()[1:])) for line in t_data_lines]).reshape(-1, 3, 4)
                lidar2cam = pad_poses(np.array(lidar2cam))
                cam2lidar = np.linalg.inv(lidar2cam)
                cam2lidars.append(cam2lidar)
            Ks_list = np.array(Ks_list)
            c2w = np.array(self.ego_poses)[:, None, ...] @ np.array(cam2lidars)
            
            for frame_idx in range(self.frame_num):
                timestamp = time_duration[0] + (time_duration[1] - time_duration[0]) * frame_idx / (self.frame_num - 1)
                for cam_idx in range(cam_num):
                    c2w_transformed = np.eye(4)
                    R_transformed = np.array(cam_data[cam_num * frame_idx + cam_idx]['rotation'])
                    T_transformed = np.array(cam_data[cam_num * frame_idx + cam_idx]['position'])
                    c2w_transformed[:3, :3] = R_transformed
                    c2w_transformed[:3, 3] = T_transformed
                    transform = c2w_transformed @ np.linalg.inv(c2w[frame_idx, cam_idx, :, :])
                    scale_factor = np.power(np.linalg.det(transform[:3, :3]), 1/3)
                    transform[:3, :3] *= scale_factor

                    R = c2w[frame_idx, cam_idx, :3, :3]
                    T = c2w[frame_idx, cam_idx, :3, 3]
                    fx = cam_data[cam_num * frame_idx + cam_idx]['fx']
                    fy = cam_data[cam_num * frame_idx + cam_idx]['fy']
                    cx = cam_data[cam_num * frame_idx + cam_idx]['cx']
                    cy = cam_data[cam_num * frame_idx + cam_idx]['cy']
                    width = cam_data[cam_num * frame_idx + cam_idx]['width']
                    height = cam_data[cam_num * frame_idx + cam_idx]['height']

                    cam_infos.setdefault(cam_idx, []).append(
                        PVGCameraInfo(
                            idx=cam_idx, timestamp=timestamp, rotation=R, position=T, 
                            fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height, 
                            transform=transform, scale_factor=scale_factor)
                    )
        else:
            cam_num = 0
            transform_path = os.path.join(self.model_path, 'camera_transform.npz')
            if os.path.exists(transform_path):
                transform_data = np.load(transform_path)
                transform = transform_data['transform']
                scale_factor = transform_data['scale_factor'].item()
            else:
                transform = np.eye(4)
                scale_factor = 1.0

            cam2lidars = []
            Ks_list = []
            for frame_idx, calib_file in enumerate(calib_list):
                with open(calib_file) as f:
                    lines = f.readlines()
                p_target_line_index = None
                t_target_line_index = None
                for i, line in enumerate(lines):
                    if "P0:" in line:
                        p_target_line_index = i
                    if "T_lidar2cam0" in line:
                        t_target_line_index = i
                    if line[0] == 'P' and frame_idx == 0:
                        cam_num += 1
                p_data_lines = lines[p_target_line_index : p_target_line_index + cam_num]
                t_data_lines = lines[t_target_line_index : t_target_line_index + cam_num]
                Ks = np.array([list(map(float, line.split()[1:])) for line in p_data_lines]).reshape(-1, 3, 4)[:, :, :3]
                Ks_list.append(Ks)
                lidar2cam = np.array([list(map(float, line.split()[1:])) for line in t_data_lines]).reshape(-1, 3, 4)
                lidar2cam = pad_poses(np.array(lidar2cam))
                cam2lidar = np.linalg.inv(lidar2cam)
                cam2lidars.append(cam2lidar)
            Ks_list = np.array(Ks_list)
            c2w = np.array(self.ego_poses)[:, None, ...] @ np.array(cam2lidars)
            for frame_idx in range(self.frame_num):
                timestamp = time_duration[0] + (time_duration[1] - time_duration[0]) * frame_idx / (self.frame_num - 1)
                for cam_idx in range(cam_num):
                    R = c2w[frame_idx, cam_idx, :3, :3]
                    T = c2w[frame_idx, cam_idx, :3, 3]
                    K = Ks_list[frame_idx, cam_idx]
                    fx = float(K[0, 0])
                    fy = float(K[1, 1])
                    cx = float(K[0, 2])
                    cy = float(K[1, 2])
                    cam_infos.setdefault(cam_idx, []).append(
                        PVGCameraInfo(
                            idx=cam_idx, timestamp=timestamp, rotation=R, position=T, 
                            fx=fx, fy=fy, cx=cx, cy=cy, width=W, height=H, 
                            transform=transform, scale_factor=scale_factor
                        )
                    ) 
        return cam_infos