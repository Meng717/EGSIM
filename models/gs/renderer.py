import torch
from basic.cameras import PinholeCamInfo
from models.gs.model_loader import GaussianInfo
from typing import NamedTuple, Literal
from gsplat import rasterization
from models.basic.utils.graphics_utils import getTransformMatrix

def getIntrMat(fx, fy, width, height):
    cx = width / 2.0
    cy = height / 2.0
    
    Ks = torch.tensor([
        [fx,  0,  cx],
        [0,  fy,  cy],
        [0,   0,  1 ] 
    ], dtype=torch.float32, device="cuda")
    
    return Ks

class RazterizerSettings(NamedTuple):
    render_mode: Literal['RGB', 'D', 'ED', 'RGB+D', 'RGB+ED']

class GaussianRenderer:
    
    def __init__(self, **kwargs):
        if 'render_mode' in kwargs:
            self._razterizer_settings = RazterizerSettings(
                render_mode=kwargs["render_mode"]
            )
    
    def set(self, **kwargs):
        self._razterizer_settings = RazterizerSettings(
            render_mode=kwargs["render_mode"]
        )
    
    def getSettings(self) -> dict:
        return self._razterizer_settings._asdict()
    
    def razterization(self, camera: PinholeCamInfo, gaussians: GaussianInfo):
        render_result, render_alpha, _ = rasterization(
            means=gaussians.means, quats=gaussians.quats,
            scales=gaussians.scales, opacities=gaussians.opacities, 
            colors=gaussians.colors, sh_degree=gaussians.sh_degree, 
            viewmats=torch.tensor(getTransformMatrix(camera.rotation.T, -camera.rotation.T @ camera.position)[None, ...], 
                                  device="cuda"),
            Ks=getIntrMat(camera.fx, camera.fy, camera.width, camera.height)[None, ...], 
            width=camera.width, height=camera.height, near_plane=camera.near, 
            far_plane=camera.far, camera_model='pinhole', 
            render_mode=self._razterizer_settings.render_mode, eps2d=0.1
        )
        render_result = render_result.detach().cpu().numpy().squeeze()
        return render_result, render_alpha