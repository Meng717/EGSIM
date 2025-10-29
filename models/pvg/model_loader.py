import os
import numpy as np
import torch
import glob
from torch import Tensor
from torch.nn.parameter import Parameter
from dataclasses import dataclass
from models.basic.basic_class import BasicModelLoader
from models.basic.utils.general_utils import build_scaling_rotation, strip_symmetric
import nvdiffrast.torch as dr

env_map_res = 1024

@dataclass
class PVGInfo:
    active_sh_degree: int
    means: Parameter
    features: Tensor
    scales: Tensor
    quats: Tensor
    opacities: Tensor
    t: Parameter
    scales_t: Tensor
    velocity: Parameter
    max_radii2D: Tensor
    T: float
    velocity_decay: float
    max_sh_degree: int = 3
    
    def get_xyz_SHM(self, t):
        a = 1/self.T * np.pi * 2
        return self.means + self.velocity*torch.sin((t-self.t)*a)/a
    def get_marginal_t(self, timestamp):
        return torch.exp(-0.5 * (self.t - timestamp) ** 2 / self.scales_t ** 2)
    def get_covariance(self, scaling_modifier=1):
        def build_covariance_from_scaling_rotation(scales, scaling_modifier, quats):
            L = build_scaling_rotation(scaling_modifier * scales, quats)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        return build_covariance_from_scaling_rotation(self.scales, scaling_modifier, self.quats)
    @property
    def get_inst_velocity(self):
        return self.velocity*torch.exp(-self.scales_t/self.T/2*self.velocity_decay)
    @property
    def get_max_sh_channels(self):
        return (self.max_sh_degree + 1) ** 2

class EnvLight(torch.nn.Module):

    def __init__(self, resolution=1024):
        super().__init__()
        self.to_opengl = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=torch.float32, device="cuda")
        self.base = torch.nn.Parameter(
            0.5 * torch.ones(6, resolution, resolution, 3, requires_grad=True),
        )

    def capture(self):
        return (
            self.base,
            self.optimizer.state_dict(),
        )
        
    def restore(self, model_args, training_args=None):
        self.base, opt_dict = model_args
        if training_args is not None:
            self.training_setup(training_args)
            self.optimizer.load_state_dict(opt_dict)
            
    def training_setup(self, training_args):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=training_args.envmap_lr, eps=1e-15)
        
    def forward(self, l):
        l = (l.reshape(-1, 3) @ self.to_opengl.T).reshape(*l.shape)
        l = l.contiguous()
        prefix = l.shape[:-1]
        if len(prefix) != 3:  # reshape to [B, H, W, -1]
            l = l.reshape(1, 1, -1, l.shape[-1])

        light = dr.texture(self.base[None, ...], l, filter_mode='linear', boundary_mode='cube')
        light = light.view(*prefix, -1)

        return light

class PVGModelLoader(BasicModelLoader):
    
    def set(self, model_path):
        super().set(None, model_path)
        self.gaussian_info: PVGInfo = None
        self.load_iteration = -1
        self.loaded_iter = None
        self.sh_degree = 3
        self.device = torch.device("cuda")

        self.scales_activation = torch.exp
        self.scales_t_activation = torch.exp
        self.opacity_activation = torch.sigmoid
        self.rotation_activation = torch.nn.functional.normalize

        if env_map_res > 0:
            self.env_map = EnvLight(resolution=env_map_res).cuda()
        else:
            self.env_map = None
    
    def load(self) -> tuple[PVGInfo, EnvLight]:
        
        checkpoints = glob.glob(os.path.join(self.model_path, "chkpnt*.pth"))
        assert len(checkpoints) > 0, "No checkpoints found."
        checkpoint = sorted(checkpoints, key=lambda x: int(x.split("chkpnt")[-1].split(".")[0]))[-1]
        (model_params, _) = torch.load(checkpoint)
        (active_sh_degree, _means, _features_dc, _features_rest, _scales,
         _quats, _opacities, _t, _scales_t, _velocity, max_radii2D,
         _, _, _, _, _, T, velocity_decay) = model_params
        self.gaussian_info = PVGInfo(
            active_sh_degree=active_sh_degree, means=_means, 
            features=torch.cat((_features_dc, _features_rest), dim=1),
            scales=self.scales_activation(_scales), quats=self.rotation_activation(_quats), 
            opacities=self.opacity_activation(_opacities), t=_t, scales_t=self.scales_t_activation(_scales_t), 
            velocity=_velocity, max_radii2D=max_radii2D, T=T, velocity_decay=velocity_decay
        )
        
        if self.env_map is not None:
            env_checkpoint = os.path.join(os.path.dirname(checkpoint), 
                                        os.path.basename(checkpoint).replace("chkpnt", "env_light_chkpnt"))
            (light_params, _) = torch.load(env_checkpoint)
            self.env_map.restore(light_params)

        return (self.gaussian_info, self.env_map)
    
    def restore(self) -> PVGInfo:
        pass
    
    def add(self, gaussians, transform: Tensor=torch.eye(4)) -> PVGInfo:
        pass
    