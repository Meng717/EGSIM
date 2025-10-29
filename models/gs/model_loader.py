import os
import numpy as np
import torch
from torch import Tensor
from typing import NamedTuple
from models.basic.utils.system_utils import searchForMaxIteration
from plyfile import PlyData
from models.basic.basic_class import BasicModelLoader

class GaussianInfo(NamedTuple):
    means: Tensor                   # [N, 3]
    quats: Tensor                   # [N, 4]
    scales: Tensor                  # [N, 3]
    opacities: Tensor               # [N]
    colors: Tensor                  # [(C,) N, D] or [(C,) N, K, 3]
    sh_degree: int

class ModelLoader(BasicModelLoader):
    
    def set(self, model_path):
        super().set(model_path)
        self.gaussian_info: GaussianInfo = None
        self.load_iteration = -1
        self.loaded_iter = None
        self.sh_degree = 3
        self.device = torch.device("cuda")

        self.scaling_activation = torch.exp
        self.opacity_activation = torch.sigmoid
        self.rotation_activation = torch.nn.functional.normalize
    
    def load(self) -> GaussianInfo:
        if self.load_iteration:
            if self.load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = self.load_iteration
        plydata = PlyData.read(os.path.join(self.model_path, "point_cloud", 
                                        f"iteration_{self.loaded_iter}", "point_cloud.ply"))
        
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis].squeeze()

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.sh_degree + 1) ** 2 - 1)).transpose(0, 2, 1)
        features_dc = features_dc.transpose(0, 2, 1)
        features = np.concatenate((features_dc, features_extra), axis=1)

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        self.gaussian_info = GaussianInfo(
            means=torch.tensor(xyz, dtype=torch.float32, device=self.device), 
            quats=self.rotation_activation(torch.tensor(rots, dtype=torch.float32, device=self.device)), 
            scales=self.scaling_activation(torch.tensor(scales, dtype=torch.float32, device=self.device)), 
            opacities=self.opacity_activation(torch.tensor(opacities, dtype=torch.float32, device=self.device)), 
            colors=torch.tensor(features, dtype=torch.float32, device=self.device),
            sh_degree=self.sh_degree
        )
        return self.gaussian_info
    
    def restore(self) -> GaussianInfo:
        model_args = torch.load(self.model_path)[0]
        self.gaussian_info = GaussianInfo(
            sh_degree=model_args[0], means=model_args[1], colors=torch.cat((model_args[2], model_args[3]), dim=1),
            scales=self.scaling_activation(model_args[5]), quats=self.rotation_activation(model_args[6]), 
            opacities=self.opacity_activation(model_args[7].squeeze())
        )
        return self.gaussian_info
    
    def add(self, gaussians, transform: Tensor=torch.eye(4)) -> GaussianInfo:
        transform = transform.to(gaussians.means.device)
        transformed_means = (torch.cat([gaussians.means, 
                             torch.ones_like(gaussians.means[..., :1])], 
                             dim=1) @ transform.T)[..., :3]
        transformed_quats = gaussians.quats @ transform.T
        self.gaussian_info = GaussianInfo(
            means=torch.cat((self.gaussian_info.means, transformed_means), dim=0), 
            quats=torch.cat((self.gaussian_info.quats, transformed_quats), dim=0), 
            scales=torch.cat((self.gaussian_info.scales, gaussians.scales), dim=0), 
            opacities=torch.cat((self.gaussian_info.opacities, gaussians.opacities), dim=0), 
            colors=torch.cat((self.gaussian_info.colors, gaussians.colors), dim=0),
            sh_degree=self.gaussian_info.sh_degree 
        )
        return self.gaussian_info