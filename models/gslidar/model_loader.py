import torch
import os
import glob
from models.basic.utils.general_utils import inverse_sigmoid
from models.basic.basic_class import BasicModelLoader
from models.pvg.model_loader import PVGInfo
from .unet import UNet
from dataclasses import dataclass

@dataclass
class GSLidarModelInfo(PVGInfo):
    intensity: float = ...

class RayDropPrior(torch.nn.Module):
    def __init__(self, h, w):
        super().__init__()
        init = inverse_sigmoid(0.1 * torch.ones([1, h, w * 2]))
        self.lidar_raydrop_prior = torch.nn.Parameter(init, requires_grad=True)

    def capture(self):
        return self.lidar_raydrop_prior

    def restore(self, model_args):
        self.lidar_raydrop_prior, _ = model_args

    def forward(self, towards):
        w = self.lidar_raydrop_prior.shape[-1] // 2
        if towards == "forward":
            lidar_raydrop_prior_from_envmap = self.lidar_raydrop_prior[:, :, :w]
        elif towards == "backward":
            lidar_raydrop_prior_from_envmap = self.lidar_raydrop_prior[:, :, w:]
        else:
            raise NotImplementedError(towards)
        return torch.sigmoid(lidar_raydrop_prior_from_envmap)

class GSLidarModelLoader(BasicModelLoader):
    
    def set(self, model_path, width=540, height=32):
        super().set(None, model_path)
        self.gaussian_info: PVGInfo = None
        self.load_iteration = -1
        self.loaded_iter = None
        self.sh_degree = 3
        self.device = torch.device("cuda")

        self.scaling_activation = torch.exp
        self.scaling_t_activation = torch.exp
        self.opacity_activation = torch.sigmoid
        self.rotation_activation = torch.nn.functional.normalize

        self.lidar_raydrop_prior = RayDropPrior(height, width)
        self.unet = UNet(in_channels=3, out_channels=1)
        self.unet.cuda()
        self.unet.eval()
    
    def load(self) -> tuple[GSLidarModelInfo, RayDropPrior, UNet]:
        
        checkpoints = glob.glob(os.path.join(self.model_path, "chkpnt*.pth"))
        assert len(checkpoints) > 0, "No checkpoints found."
        checkpoint = sorted(checkpoints, key=lambda x: int(x.split("chkpnt")[-1].split(".")[0]))[-1]
        (model_params, _) = torch.load(checkpoint)
        (active_sh_degree, _xyz, _features_dc, _features_rest, _scaling,
         _rotation, _opacity, _t, _scaling_t, _velocity, _intensity, max_radii2D,
         _, _, _, _, _, _, T, velocity_decay) = model_params
        self.gaussian_info = GSLidarModelInfo(
            active_sh_degree=active_sh_degree, means=_xyz, 
            features=torch.cat((_features_dc, _features_rest), dim=1),
            scales=self.scaling_activation(_scaling), quats=self.rotation_activation(_rotation), 
            opacities=self.opacity_activation(_opacity), t=_t, scales_t=self.scaling_t_activation(_scaling_t), 
            velocity=_velocity, max_radii2D=max_radii2D, T=T, velocity_decay=velocity_decay, intensity=_intensity
        )
        
        raydrop_prior_checkpoint = os.path.join(os.path.dirname(checkpoint), 
                                                os.path.basename(checkpoint).replace("chkpnt", "lidar_raydrop_prior_chkpnt"))
        (lidar_raydrop_prior_params, _) = torch.load(raydrop_prior_checkpoint)
        self.lidar_raydrop_prior.restore(lidar_raydrop_prior_params)

        refine_path = os.path.join(self.model_path, "refine.pth")
        self.unet.load_state_dict(torch.load(refine_path))
        return (self.gaussian_info, self.lidar_raydrop_prior, self.unet)
