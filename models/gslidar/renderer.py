import torch
import numpy as np
import math
from dataclasses import dataclass
from models.gslidar.lidar_loader import GSLidarInfo
from models.gslidar.model_loader import GSLidarModelInfo, RayDropPrior, UNet, GSLidarModelLoader
from models.gslidar.diff_gaussian_rasterization_2d import GaussianRasterizationSettings, GaussianRasterizer
from models.basic.utils.sh_utils import eval_sh
from typing import Literal
import torch.nn.functional as F
from configs.constants import CoordTransform

@dataclass
class RazterizerSettings:
    override_color = None
    mask = None
    dynamic: bool = False
    time_shift: float = None   
    white_background: bool = False
    scaling_modifier: float = 1.0
    neg_fov: bool = True
    compute_cov3D_python: bool = False
    convert_SHs_python: bool = False    
    debug: bool = False
    median_depth: bool = False
    sky_depth: bool = False
    depth_blend_mode = 0
    other = [] 

def pano_to_lidar(range_image, vfov, hfov):
    mask = range_image > 0

    panorama_height, panorama_width = range_image.shape[-2:]
    theta, phi = torch.meshgrid(torch.arange(panorama_height, device=range_image.device),
                                torch.arange(panorama_width, device=range_image.device), indexing="ij")

    vertical_degree_range = vfov[1] - vfov[0]
    theta = (90 - vfov[1] + theta / panorama_height * vertical_degree_range) * torch.pi / 180

    horizontal_degree_range = hfov[1] - hfov[0]
    phi = (hfov[0] + phi / panorama_width * horizontal_degree_range) * torch.pi / 180

    dx = torch.sin(theta) * torch.sin(phi)
    dz = torch.sin(theta) * torch.cos(phi)
    dy = -torch.cos(theta)

    directions = torch.stack([dx, dy, dz], dim=0)
    directions = F.normalize(directions, dim=0)

    points_xyz = (directions * range_image)[:, mask[0]].permute(1, 0)

    return points_xyz

def render(viewpoint_camera: GSLidarInfo, pc: GSLidarModelInfo, env_map: RayDropPrior, settings: RazterizerSettings, 
           towards: Literal['forward', 'backward']):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros((pc.means.shape[0], 4), dtype=pc.means.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    tanfovx = math.tan(-0.5)
    tanfovy = math.tan(-0.5)

    bg_color = [1, 1, 1, 1] if settings.white_background else [0, 0, 0, 1]  # 无穷远的ray drop概率为1
    bg_color = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.height),
        image_width=int(viewpoint_camera.width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color if env_map is not None else torch.zeros(3, device="cuda"),
        scale_modifier=settings.scaling_modifier,
        viewmatrix=torch.tensor(viewpoint_camera.get_w2c()[towards]).transpose(0, 1).cuda(),
        projmatrix=torch.tensor(viewpoint_camera.get_w2c()[towards]).transpose(0, 1).cuda(),
        sh_degree=pc.active_sh_degree,
        campos=torch.tensor(viewpoint_camera.get_center()[towards], dtype=torch.float32, device="cuda"),
        prefiltered=False,
        debug=settings.debug,
        vfov=viewpoint_camera.vfov,
        hfov=viewpoint_camera.hfov,
        scale_factor=viewpoint_camera.scale_factor
    )

    assert raster_settings.bg.shape[0] == 4

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means2D = screenspace_points
    opacity = pc.opacities
    scales = None
    rotations = None
    cov3D_precomp = None

    if settings.time_shift is not None:
        means3D = pc.get_xyz_SHM(viewpoint_camera.timestamp - settings.time_shift)
        means3D = means3D + pc.get_inst_velocity * settings.time_shift
        marginal_t = pc.get_marginal_t(viewpoint_camera.timestamp - settings.time_shift)
    else:
        means3D = pc.get_xyz_SHM(viewpoint_camera.timestamp)
        marginal_t = pc.get_marginal_t(viewpoint_camera.timestamp)

    # 静态不乘 marginal_t 了
    if settings.dynamic:
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
            dir_pp = (means3D.detach() - viewpoint_camera.get_center()[towards].repeat(pc.features.shape[0], 1)).detach()
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
    mask = (opacity[:, 0] > 1 / 255) if settings.mask is None else mask & (opacity[:, 0] > 1 / 255)
    if settings.dynamic:
        mask = mask & (marginal_t[:, 0] > 0.05)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    contrib, rendered_image, rendered_feature, rendered_depth, rendered_opacity, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        features=features,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
        mask=mask)

    _, rendered_intensity_sh, rendered_raydrop = rendered_image.split([2, 1, 1], dim=0)
    rendered_other, rendered_normal = rendered_feature.split([S_other, 3], dim=0)
    rendered_normal = rendered_normal / (rendered_normal.norm(dim=0, keepdim=True) + 1e-8)

    if env_map is not None:
        lidar_raydrop_prior_from_envmap = env_map(towards)
        rendered_raydrop = lidar_raydrop_prior_from_envmap + (1 - lidar_raydrop_prior_from_envmap) * rendered_raydrop

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "contrib": contrib,
        "depth": rendered_depth[[1]] if settings.median_depth else rendered_depth[[0]],
        "depth_mean": rendered_depth[[0]],
        "depth_median": rendered_depth[[1]],
        "distortion": rendered_depth[[2]],
        "depth_square": rendered_depth[[3]],
        "alpha": rendered_opacity,
        "feature": rendered_other,
        "normal": rendered_normal,
        "intensity_sh": rendered_intensity_sh,
        "raydrop": rendered_raydrop.clamp(0, 1)
    }

def render_range_map(lidar: GSLidarInfo, gaussians: GSLidarModelInfo, env_map: RayDropPrior, settings: RazterizerSettings):

    EPS = 1e-5
    h, w = lidar.height, lidar.width
    breaks = (0, w // 2, 3 * w // 2, w * 2)

    depth_pano = torch.zeros([3, h, w * 2]).cuda()
    intensity_sh_pano = torch.zeros([1, h, w * 2]).cuda()
    raydrop_pano = torch.zeros([1, h, w * 2]).cuda()

    for viewpoint in ['forward', 'backward']:
        render_pkg = render(lidar, gaussians, env_map, settings, viewpoint)

        depth = render_pkg['depth']
        alpha = render_pkg['alpha']
        raydrop_render = render_pkg['raydrop']

        depth_var = render_pkg['depth_square'] - depth ** 2
        depth_median = render_pkg["depth_median"]
        var_quantile = depth_var.median() * 10

        depth_mix = torch.zeros_like(depth)
        depth_mix[depth_var > var_quantile] = depth_median[depth_var > var_quantile]
        depth_mix[depth_var <= var_quantile] = depth[depth_var <= var_quantile]

        depth = torch.cat([depth_mix, depth, depth_median])

        if settings.sky_depth:
            sky_depth = 900
            depth = depth / alpha.clamp_min(EPS)
            if settings.depth_blend_mode == 0:  # harmonic mean
                depth = 1 / (alpha / depth.clamp_min(EPS) + (1 - alpha) / sky_depth).clamp_min(EPS)
            elif settings.depth_blend_mode == 1:
                depth = alpha * depth + (1 - alpha) * sky_depth

        intensity_sh = render_pkg['intensity_sh']

        if viewpoint == 'forward':  # 前180度
            depth_pano[:, :, breaks[1]:breaks[2]] = depth
            intensity_sh_pano[:, :, breaks[1]:breaks[2]] = intensity_sh
            raydrop_pano[:, :, breaks[1]:breaks[2]] = raydrop_render
            continue
        elif viewpoint == 'backward':
            depth_pano[:, :, breaks[2]:breaks[3]] = depth[:, :, 0:(breaks[3] - breaks[2])]
            depth_pano[:, :, breaks[0]:breaks[1]] = depth[:, :, (w - breaks[1] + breaks[0]):w]

            intensity_sh_pano[:, :, breaks[2]:breaks[3]] = intensity_sh[:, :, 0:(breaks[3] - breaks[2])]
            intensity_sh_pano[:, :, breaks[0]:breaks[1]] = intensity_sh[:, :, (w - breaks[1] + breaks[0]):w]

            raydrop_pano[:, :, breaks[2]:breaks[3]] = raydrop_render[:, :, 0:(breaks[3] - breaks[2])]
            raydrop_pano[:, :, breaks[0]:breaks[1]] = raydrop_render[:, :, (w - breaks[1] + breaks[0]):w]

    return depth_pano, intensity_sh_pano, raydrop_pano

class GSLidarRenderer:
    def __init__(self, **kwargs):
        self._razterizer_settings = RazterizerSettings()
    
    def set(self, **kwargs):
        pass
    
    def getSettings(self) -> dict:
        return self._razterizer_settings.__dict__
    
    def razterization(self, lidar: GSLidarInfo, model_info: tuple[GSLidarModelInfo, RayDropPrior, UNet]) -> np.ndarray:
        with torch.no_grad():
            depth_pano, intensity_sh_pano, raydrop_pano \
                = render_range_map(lidar, model_info[0], model_info[1], self._razterizer_settings)
            test_input = torch.cat([raydrop_pano, intensity_sh_pano, depth_pano[[0]]]).cuda().float().contiguous().unsqueeze(0)
            unet = model_info[2]
            raydrop_refine = unet(test_input)
            raydrop_mask = torch.where(raydrop_refine > 0.5, 1, 0)
            raydrop_pano = raydrop_refine[0, [0]]
            raydrop_pano_mask = raydrop_mask[0, [0]]
            # intensity_pano = test_input[0, [1]] * (1 - raydrop_pano_mask)
            depth_pano = test_input[0, [2]] * (1 - raydrop_pano_mask)

            points = pano_to_lidar(depth_pano, lidar.vfov, (-180, 180)).cpu().numpy() / lidar.scale_factor
        return points @ CoordTransform.camera2lidar[:3, :3].T
