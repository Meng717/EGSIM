import os
import numpy as np
import glob
from models.basic.cameras import PinholeCamInfo
from models.basic.utils.general_utils import pad_poses
from models.basic.basic_class import BasicSensorLoader

H = 900
W = 1600
    
class CameraLoader(BasicSensorLoader):

    def load(self) -> dict[PinholeCamInfo]:
        cam_num = 0
        cam_infos = {}
        calib_list = sorted(glob.glob(os.path.join(self.model_path, 'calib', '*.txt')))
        transform_path = os.path.join(self.model_path, 'render', 'camera_render_model', 'camera_transform.npz')
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
            for cam_idx in range(cam_num):
                R = c2w[frame_idx, cam_idx, :3, :3]
                T = c2w[frame_idx, cam_idx, :3, 3]
                K = Ks_list[frame_idx, cam_idx]
                fx = float(K[0, 0])
                fy = float(K[1, 1])
                cx = float(K[0, 2])
                cy = float(K[1, 2])
                cam_infos.setdefault(cam_idx, []).append(
                    PinholeCamInfo(
                        idx=cam_idx, rotation=R, position=T, 
                        fx=fx, fy=fy, cx=cx, cy=cy, width=W, height=H, 
                        transform=transform, scale_factor=scale_factor
                    )
                ) 
        return cam_infos
