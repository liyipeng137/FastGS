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
import math
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from diff_gaussian_rasterization_fastgs import GaussianRasterizationSettings, GaussianRasterizer

def render_fastgs(
    viewpoint_camera,
    pc : GaussianModel,
    pipe,
    bg_color : torch.Tensor,
    mult,
    scaling_modifier = 1.0,
    override_color = None,
    get_flag=None,
    metric_map = None,
    render_mask = None,
):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    means3D = pc.get_xyz
    features_dc = pc.get_features_dc
    features_rest = pc.get_features_rest
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation

    if render_mask is not None:
        render_mask = render_mask.to(device=means3D.device, dtype=torch.bool)
        means3D = means3D[render_mask]
        features_dc = features_dc[render_mask]
        features_rest = features_rest[render_mask]
        opacity = opacity[render_mask]
        scales = scales[render_mask]
        rotations = rotations[render_mask]
        if override_color is not None and override_color.shape[0] == render_mask.shape[0]:
            override_color = override_color[render_mask]

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    # screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points = torch.zeros((means3D.shape[0], 4), dtype=means3D.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    if metric_map==None:
        metric_map=torch.zeros(int(viewpoint_camera.image_height)*int(viewpoint_camera.image_width), dtype=torch.int, device='cuda')

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        mult = mult,
        prefiltered=False,
        debug=pipe.debug,
        get_flag=get_flag,
        metric_map = metric_map
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means2D = screenspace_points

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        if render_mask is None:
            cov3D_precomp = pc.get_covariance(scaling_modifier)
        else:
            cov3D_precomp = pc.get_covariance(scaling_modifier)[render_mask]
        scales = None
        rotations = None

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    dc = None
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            features = torch.cat((features_dc, features_rest), dim=1)
            shs_view = features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (means3D - viewpoint_camera.camera_center.repeat(features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            dc, shs = features_dc, features_rest
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, accum_metric_counts = rasterizer(
        means3D = means3D,
        means2D = means2D,
        dc = dc,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : (radii > 0).nonzero(),
            "radii": radii,
            "accum_metric_counts" : accum_metric_counts}

def render_gsplat_depth(viewpoint_camera, pc: GaussianModel, render_mode="ED", include_background=False):
    """Render expected z-depth with gsplat while keeping FastGS rendering unchanged."""
    try:
        from gsplat import rasterization
    except ImportError as exc:
        raise ImportError(
            "gsplat is required for depth prior supervision. Install gsplat or "
            "leave depth_prior_dir empty to disable depth loss."
        ) from exc

    height = int(viewpoint_camera.image_height)
    width = int(viewpoint_camera.image_width)
    device = pc.get_xyz.device
    dtype = pc.get_xyz.dtype
    render_mask = None
    if not include_background and hasattr(pc, "get_foreground_mask"):
        render_mask = pc.get_foreground_mask

    means = pc.get_xyz
    quats = pc.get_rotation
    scales = pc.get_scaling
    opacities = pc.get_opacity.squeeze(-1)
    if render_mask is not None:
        render_mask = render_mask.to(device=device, dtype=torch.bool)
        means = means[render_mask]
        quats = quats[render_mask]
        scales = scales[render_mask]
        opacities = opacities[render_mask]

    if means.shape[0] == 0:
        return {
            "depth": torch.zeros((1, height, width), dtype=dtype, device=device),
            "alpha": torch.zeros((1, height, width), dtype=dtype, device=device),
            "meta": {},
        }

    fx = width / (2.0 * math.tan(viewpoint_camera.FoVx * 0.5))
    fy = height / (2.0 * math.tan(viewpoint_camera.FoVy * 0.5))
    K = torch.tensor(
        [[fx, 0.0, width * 0.5], [0.0, fy, height * 0.5], [0.0, 0.0, 1.0]],
        dtype=dtype,
        device=device,
    )[None]
    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1).to(
        device=device,
        dtype=dtype,
    )[None].contiguous()

    renders, alphas, meta = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=None,
        viewmats=viewmat,
        Ks=K,
        width=width,
        height=height,
        near_plane=viewpoint_camera.znear,
        far_plane=viewpoint_camera.zfar,
        render_mode=render_mode,
    )

    return {
        "depth": renders[0, ..., 0].unsqueeze(0),
        "alpha": alphas[0, ..., 0].unsqueeze(0),
        "meta": meta,
    }
