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

from scene.cameras import Camera
import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal

WARNED = False

def _resolve_prior_root(source_path, prior_dir):
    prior_dir = "" if prior_dir is None else str(prior_dir).strip()
    if not prior_dir:
        return None
    if os.path.isabs(prior_dir):
        return prior_dir
    return os.path.join(source_path, prior_dir)

def _load_depth_prior(args, cam_info, resolution):
    depth_prior_dir = getattr(args, "depth_prior_dir", "")
    depth_root = _resolve_prior_root(args.source_path, depth_prior_dir)
    if depth_root is None:
        return None

    depth_format = (getattr(args, "depth_prior_format", "png") or "png").lower().lstrip(".")
    depth_path = os.path.join(depth_root, f"{cam_info.image_name}.{depth_format}")
    if not os.path.exists(depth_path):
        return None

    depth_img = Image.open(depth_path)
    depth_np = np.asarray(depth_img).astype(np.float32)
    if depth_np.ndim == 3:
        depth_np = depth_np[..., 0]

    depth_scale = float(getattr(args, "depth_prior_scale", 1000.0))
    depth_prior = torch.from_numpy(depth_np)[None, None] / depth_scale
    depth_prior = F.interpolate(
        depth_prior,
        size=(resolution[1], resolution[0]),
        mode="nearest",
    )[0]
    return depth_prior

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    depth_prior = _load_depth_prior(args, cam_info, resolution)

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device,
                  depth_prior=depth_prior)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
