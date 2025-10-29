import os
import numpy as np
import torch
import torch.nn.functional as F
from models.basic.lidars import LidarInfo
from models.basic.basic_class import BasicSensorLoader
from dataclasses import dataclass
import camtools as ct
from configs.constants import CoordTransform

time_duration = [-0.5, 0.5]
vfov = [-30, 10]
hfov = [-90, 90]
width = 540
height = 32
lidar2cam = np.linalg.inv(CoordTransform.camera2lidar)

def normalize_Ts(Ts):
    # New Cs.
    Cs = np.array([ct.convert.T_to_C(T) for T in Ts])
    normalize_mat = ct.normalize.compute_normalize_mat(Cs)
    # Cs_new = ct.project.homo_project(Cs.reshape((-1, 3)), normalize_mat)
    Cs_new = (np.concatenate([Cs, np.ones((Cs.shape[0], 1))], axis=1) @ normalize_mat.T)[:, :3]

    # New Ts.
    Ts_new = []
    for T, C_new in zip(Ts, Cs_new):
        pose = ct.convert.T_to_pose(T)
        pose[:3, 3] = C_new
        T_new = ct.convert.pose_to_T(pose)
        Ts_new.append(T_new)

    return Ts_new

@dataclass
class GSLidarInfo(LidarInfo):
    timestamp: float = ...

    def get_c2w(self):
        return {
            'forward': (self.transformed_c2w.cpu().numpy() @ np.linalg.inv(lidar2cam)).astype(np.float32), 
            'backward': (self.transformed_c2w.cpu().numpy() @ np.linalg.inv(lidar2cam) \
                @ np.array([-1, 0, 0, 0,
                            0, 1, 0, 0,
                            0, 0, -1, 0,
                            0, 0, 0, 1]).reshape(4, 4)).astype(np.float32)
        }
    def get_w2c(self):
        return {
            'forward': np.linalg.inv(self.get_c2w()['forward']),
            'backward': np.linalg.inv(self.get_c2w()['backward'])
        }
    def get_center(self):
        return {
            'forward': self.get_w2c()['forward'][:3, 3], 
            'backward': self.get_w2c()['backward'][:3, 3]
        }
    
    def get_world_directions_panorama(self):
        theta, phi = torch.meshgrid(torch.arange(self.height, device='cuda'),
                                    torch.arange(self.width, device='cuda'), indexing="ij")

        vertical_degree_range = self.vfov[1] - self.vfov[0]
        theta = (90 - self.vfov[1] + theta / self.height * vertical_degree_range) * torch.pi / 180

        horizontal_degree_range = self.hfov[1] - self.hfov[0]
        phi = (self.hfov[0] + phi / self.width * horizontal_degree_range) * torch.pi / 180

        dx = torch.sin(theta) * torch.sin(phi)
        dz = torch.sin(theta) * torch.cos(phi)
        dy = -torch.cos(theta)

        directions = torch.stack([dx, dy, dz], dim=0)
        directions = F.normalize(directions, dim=0)
        directions = (self.c2w[:3, :3] @ directions.reshape(3, -1)).reshape(3, self.height,
                                                                            self.width)
        return directions

    def get_local_directions_panorama(self):
        theta, phi = torch.meshgrid(torch.arange(self.height, device='cuda'),
                                    torch.arange(self.width, device='cuda'), indexing="ij")

        vertical_degree_range = self.vfov[1] - self.vfov[0]
        theta = (90 - self.vfov[1] + theta / self.height * vertical_degree_range) * torch.pi / 180

        horizontal_degree_range = self.hfov[1] - self.hfov[0]
        phi = (self.hfov[0] + phi / self.width * horizontal_degree_range) * torch.pi / 180

        dx = torch.sin(theta) * torch.sin(phi)
        dz = torch.sin(theta) * torch.cos(phi)
        dy = -torch.cos(theta)

        front_directions = torch.stack([dx, dy, dz], dim=0)
        back_directions = torch.stack([-dx, dy, -dz], dim=0)
        front_directions = F.normalize(front_directions, dim=0)
        back_directions = F.normalize(back_directions, dim=0)
        return (front_directions, back_directions)

class GSLidarLoader(BasicSensorLoader):
    
    def load(self) -> dict[LidarInfo]:
        
        lidar_infos = {}
        transformed_lidar2worlds = normalize_Ts(self.ego_poses)
        normalization_transforms = np.array(transformed_lidar2worlds) @ np.linalg.inv(np.array(self.ego_poses))

        for frame_idx in range(self.frame_num):
            sensor_idx = 0
            timestamp = time_duration[0] + (time_duration[1] - time_duration[0]) * frame_idx / (self.frame_num - 1)
            c2w = self.ego_poses[frame_idx]
            
            R = c2w[:3, :3]
            T = c2w[:3, 3]
            lidar_infos.setdefault(sensor_idx, []).append(GSLidarInfo(
                idx=sensor_idx, rotation=R, position=T, vfov=vfov, hfov=hfov, 
                width=width, height=height, timestamp=timestamp
            ))

        for key in lidar_infos:
            c2ws = np.zeros((len(lidar_infos[key]), 4, 4))
            Rs = np.stack([c.rotation for c in lidar_infos[key]], axis=0)
            Ts = np.stack([c.position for c in lidar_infos[key]], axis=0)
            c2ws[:, :3, :3] = Rs
            c2ws[:, :3, 3] = Ts
            c2ws[:, 3, 3] = 1

        transform_path = os.path.join(self.model_path, 'lidar_transform.npz')
        if os.path.exists(transform_path):
            transform_data = np.load(transform_path)
            transform = transform_data['transform']
            scale_factor = transform_data['scale_factor'].item()
        else:
            transform = np.eye(4)
            scale_factor = 1.0

        for frame_idx, lidar_info in enumerate(lidar_infos[key]):
            lidar_info.transform = transform @ normalization_transforms[frame_idx]
            lidar_info.scale_factor = scale_factor

        return lidar_infos
