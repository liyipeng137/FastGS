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

import os
import random
import json
import math
import numpy as np
import torch
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.graphics_utils import BasicPointCloud, getWorld2View2


def filter_point_cloud_by_scene_bounds(pcd, center, extent, dist_mult):
    points = np.asarray(pcd.points)
    dists = np.linalg.norm(points - center, axis=1)
    keep_mask = dists <= dist_mult * extent
    n_removed = int(np.sum(~keep_mask))
    if n_removed > 0:
        print(f"Filtered {n_removed} / {len(points)} points beyond {dist_mult:.2f}x scene extent")
    return BasicPointCloud(
        points=points[keep_mask],
        colors=np.asarray(pcd.colors)[keep_mask],
        normals=np.asarray(pcd.normals)[keep_mask],
    )


def get_camera_centers(cam_infos):
    centers = []
    for cam in cam_infos:
        w2c = getWorld2View2(cam.R, cam.T)
        c2w = np.linalg.inv(w2c)
        centers.append(c2w[:3, 3])
    return np.stack(centers, axis=0)


def build_background_sphere(pcd, train_cameras, args):
    n_points = int(getattr(args, "background_sphere_points", 0))
    if pcd is None or n_points <= 0:
        return None, None

    points = np.asarray(pcd.points, dtype=np.float32)
    center_mode = getattr(args, "background_sphere_center", "points")
    if center_mode == "points":
        scene_center = points.mean(axis=0)
        scene_radius = np.percentile(np.linalg.norm(points - scene_center, axis=-1), 99.9).item()
    elif center_mode == "cameras":
        camera_centers = get_camera_centers(train_cameras)
        scene_center = camera_centers.mean(axis=0)
        camera_radius = np.linalg.norm(camera_centers - scene_center, axis=-1).max().item()
        point_radius = np.percentile(np.linalg.norm(points - scene_center, axis=-1), 99.9).item()
        scene_radius = max(camera_radius, point_radius)
    else:
        raise ValueError(f"Unknown background_sphere_center: {center_mode}")

    if scene_radius <= 0.0:
        return None, None

    if n_points == 1:
        unit_sphere_points = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    else:
        samples = np.arange(n_points, dtype=np.float32)
        y = 1.0 - (samples / float(n_points - 1)) * 2.0
        radius = np.sqrt(np.clip(1.0 - y * y, 0.0, 1.0))
        phi = math.pi * (math.sqrt(5.0) - 1.0)
        theta = phi * samples
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        unit_sphere_points = np.stack([x, y, z], axis=1).astype(np.float32)

    distance = float(getattr(args, "background_sphere_distance", 2.2))
    sphere_radius = scene_radius * distance
    background_xyz = unit_sphere_points * sphere_radius + scene_center

    min_altitude = float(getattr(args, "background_sphere_min_altitude", -float("inf")))
    background_xyz = background_xyz[background_xyz[:, 2] >= min_altitude]
    if background_xyz.shape[0] == 0:
        return None, None

    color_mode = getattr(args, "background_sphere_color", "random")
    if color_mode == "random":
        background_rgb = np.random.random((background_xyz.shape[0], 3)).astype(np.float32)
    elif color_mode == "white":
        background_rgb = np.ones((background_xyz.shape[0], 3), dtype=np.float32)
    else:
        raise ValueError(f"Unknown background_sphere_color: {color_mode}")

    background_pcd = BasicPointCloud(
        points=background_xyz.astype(np.float32),
        colors=background_rgb,
        normals=np.zeros_like(background_xyz, dtype=np.float32),
    )
    prune_extent = sphere_radius * 1.0001
    print(
        "Added {} background sphere points, scene_center={}, scene_radius={}, prune_extent={}".format(
            background_xyz.shape[0],
            scene_center.tolist(),
            scene_radius,
            prune_extent,
        )
    )
    return background_pcd, prune_extent


def append_background_sphere(pcd, background_pcd):
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    normals = np.asarray(pcd.normals)
    background_points = np.asarray(background_pcd.points)
    background_colors = np.asarray(background_pcd.colors)
    background_normals = np.asarray(background_pcd.normals)
    combined_pcd = BasicPointCloud(
        points=np.concatenate([points, background_points], axis=0),
        colors=np.concatenate([colors, background_colors], axis=0),
        normals=np.concatenate([normals, background_normals], axis=0),
    )
    background_mask = np.concatenate(
        [
            np.zeros(points.shape[0], dtype=bool),
            np.ones(background_points.shape[0], dtype=bool),
        ],
        axis=0,
    )
    return combined_pcd, background_mask


class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0],
                 scene_init_dist_mult=None, scene_prune_dist_mult=None):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        self.prune_extent = self.cameras_extent
        scene_center = -np.array(scene_info.nerf_normalization["translate"], dtype=np.float32)
        self.scene_center = torch.tensor(scene_center, dtype=torch.float32, device="cuda")
        self.scene_prune_dist_mult = scene_prune_dist_mult

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            pcd = scene_info.point_cloud
            background_mask = None
            if pcd is not None and scene_init_dist_mult is not None and scene_init_dist_mult > 0:
                pcd = filter_point_cloud_by_scene_bounds(
                    pcd, scene_center, self.cameras_extent, scene_init_dist_mult)
            if getattr(args, "add_background_sphere", False):
                background_pcd, prune_extent = build_background_sphere(pcd, scene_info.train_cameras, args)
                if background_pcd is not None:
                    pcd, background_mask = append_background_sphere(pcd, background_pcd)
                    self.prune_extent = prune_extent
            self.gaussians.create_from_pcd(
                pcd,
                self.cameras_extent,
                background_mask=background_mask,
                background_opacity=getattr(args, "background_sphere_opacity", 0.99),
            )

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
