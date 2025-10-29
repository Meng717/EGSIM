#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import Tensor
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from typing import Literal
from dataclasses import dataclass
from models.pvg.camera_loader import PVGCameraInfo
from models.pvg.model_loader import PVGInfo, EnvLight
from models.basic.utils.sh_utils import eval_sh

@dataclass
class RazterizerSettings:
    render_mode: Literal['RGB', 'D']
    override_color = None
    env_map = None
    mask = None    
    time_shift: float = None   
    bg_color: Tensor = torch.tensor([0, 0, 0], 
                                    dtype=torch.float32, device="cuda")
    scaling_modifier: float = 1.0
    neg_fov: bool = True
    compute_cov3D_python: bool = False
    convert_SHs_python: bool = False    
    debug: bool = False
    other = [] 

def render(viewpoint_camera: PVGCameraInfo, pc: PVGInfo, env_map: EnvLight, settings: RazterizerSettings):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.means, dtype=pc.means.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    if settings.neg_fov:
        # we find that set fov as -1 slightly improves the results
        tanfovx = math.tan(-0.5)
        tanfovy = math.tan(-0.5)
    else:
        tanfovx = math.tan(viewpoint_camera.fovx * 0.5)
        tanfovy = math.tan(viewpoint_camera.fovy * 0.5)
    
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.height),
        image_width=int(viewpoint_camera.width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=settings.bg_color if env_map is not None else torch.zeros(3, device="cuda"),
        scale_modifier=settings.scaling_modifier,
        viewmatrix=viewpoint_camera.transformed_w2c.transpose(0, 1).cuda(),
        projmatrix=viewpoint_camera.transformed_full_proj_transform.transpose(0, 1).cuda(), 
        sh_degree=pc.active_sh_degree,
        campos=torch.tensor(viewpoint_camera.transformed_position, dtype=torch.float32, device="cuda"),
        prefiltered=False,
        debug=settings.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.means
    means2D = screenspace_points
    opacity = pc.opacities
    scales = None
    rotations = None
    cov3D_precomp = None

    if settings.time_shift is not None:
        means3D = pc.get_xyz_SHM(viewpoint_camera.timestamp-settings.time_shift)
        means3D = means3D + pc.get_inst_velocity * settings.time_shift
        marginal_t = pc.get_marginal_t(viewpoint_camera.timestamp-settings.time_shift)
    else:
        means3D = pc.get_xyz_SHM(viewpoint_camera.timestamp)
        marginal_t = pc.get_marginal_t(viewpoint_camera.timestamp)
    opacity = opacity * marginal_t

    if settings.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(settings.scaling_modifier)
    else:
        scales = pc.scales
        rotations = pc.quats

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if settings.override_color is None:
        if settings.convert_SHs_python:
            shs_view = pc.features.transpose(1, 2).view(-1, 3, pc.get_max_sh_channels)
            dir_pp = (means3D.detach() - viewpoint_camera.transformed_position.repeat(pc.features.shape[0], 1)).detach()
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.features
    else:
        colors_precomp = settings.override_color

    feature_list = settings.other

    if len(feature_list) > 0:
        features = torch.cat(feature_list, dim=1)
        S_other = features.shape[1]
    else:
        features = torch.zeros_like(means3D[:, :0])
        S_other = 0
    
    # Prefilter
    if settings.mask is None:
        mask = marginal_t[:, 0] > 0.05
    else:
        mask = mask & (marginal_t[:, 0] > 0.05)
    masked_means3D = means3D[mask]
    masked_xyz_homo = torch.cat([masked_means3D, torch.ones_like(masked_means3D[:, :1])], dim=1)
    masked_depth = (masked_xyz_homo @ viewpoint_camera.transformed_w2c.transpose(0, 1).cuda()[:, 2:3])
    depth_alpha = torch.zeros(means3D.shape[0], 2, dtype=torch.float32, device=means3D.device)
    depth_alpha[mask] = torch.cat([
        masked_depth,
        torch.ones_like(masked_depth)
    ], dim=1)
    features = torch.cat([features, depth_alpha], dim=1)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    contrib, rendered_image, rendered_feature, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        features = features,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
        mask = mask)
    
    rendered_other, rendered_depth, rendered_opacity = rendered_feature.split([S_other, 1, 1], dim=0)
    rendered_image_before = rendered_image
    if env_map is not None:
        bg_color_from_envmap = env_map(viewpoint_camera.get_world_directions.permute(1, 2, 0)).permute(2, 0, 1)
        rendered_image = rendered_image + (1 - rendered_opacity) * bg_color_from_envmap

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    if settings.render_mode == 'RGB':
        return rendered_image.permute(1,2,0).detach().cpu().numpy(), rendered_opacity
    elif settings.render_mode == 'D':
        return rendered_depth.squeeze().detach().cpu().numpy(), rendered_opacity

class PVGRenderer:
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
        return self._razterizer_settings.__dict__
    
    def razterization(self, camera: PVGCameraInfo, model_info: tuple[PVGInfo, EnvLight]):
        render_result = render(camera, model_info[0], model_info[1], self._razterizer_settings)
        return render_result