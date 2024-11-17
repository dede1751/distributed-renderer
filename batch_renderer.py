import blenderproc as bproc
from blenderproc.python.camera import CameraUtility

import os
import sys
sys.path.insert(0, os.path.abspath("./")) # Blenderproc sets PYTHONPATH independently

import urllib
import json
import math
import argparse
from typing import List, Tuple, Dict
from types import SimpleNamespace
import shutil
from dataclasses import dataclass
import logging
import random

import matplotlib.pyplot as plt

import bpy
import numpy as np
import cv2
import trimesh
from PIL import Image

from utils.writer import TarWriter

@dataclass
class BlenderScene:
    """Keep track of all elements in the scene."""
    objects: Dict[str, bproc.types.MeshObject] # objaverse_id -> MeshObject
    planes: List[bproc.types.MeshObject]
    light_plane: bproc.types.MeshObject
    light_plane_material: bproc.types.Material
    light_points: List[bproc.types.MeshObject]
    hdri_files: List[str]

def setup_blender(data_path, objects, cfg):
    bproc.init()

    # Initialize fixed components of the scene and load all objects
    loaded_objects = load_objects(data_path, objects)

    if cfg.use_hdri:
        hdri_files = get_hdri_files(cfg.hdri_path)
        scene = BlenderScene(
            loaded_objects, None,
            None, None, None, hdri_files)
    else:
        planes = add_bowl()
        light_plane, light_plane_material, light_points = add_lights()
        scene = BlenderScene(
            loaded_objects, planes, 
            light_plane, light_plane_material, light_points, None)
    
    if cfg.generate_pcd:
        # Add uniformly distributed cameras on a sphere
        for i in range(cfg.pcd.num_views):
            location = bproc.sampler.sphere(center=[0, 0, 0], radius=cfg.pcd.cam_dist, mode='SURFACE')
            rotation_matrix = bproc.camera.rotation_from_forward_vec(-location)
            cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
            bproc.camera.add_camera_pose(cam2world_matrix, frame=cfg.cam.num_views + i)

    # Initialize renderer settings
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    bproc.renderer.set_max_amount_of_samples(cfg.num_samples)
    bproc.renderer.set_output_format(enable_transparency=cfg.remove_background)
    #bproc.renderer.set_cpu_threads(2)

    return scene

def load_objects(data_path, objects):
    loaded = {}

    for obj in objects:
        obj_path = os.path.join(data_path, "objs", obj["github_id"], obj["objaverse_id"], "model_scaled.obj")

        blender_objs = bproc.loader.load_obj(obj_path)
        if len(blender_objs) > 1:
            logging.error(f"Loaded multiple objects at once from: {obj_path}")
        loaded_obj = blender_objs[-1]
        
        loaded_obj.set_shading_mode('auto')
        loaded_obj.hide(True)
        loaded[obj["objaverse_id"]]= loaded_obj

    return loaded

def get_hdri_files(hdri_base_path):
    hdri_files = []
    for root, dirs, files in os.walk(hdri_base_path):
        for file in files:
            if file.endswith('.exr') or file.endswith('.hdr'):
                hdri_files.append(os.path.join(root, file))
    return hdri_files

def add_bowl():
    """Add rectangular bowl to the scene."""
    planes = [
        bproc.object.create_primitive('PLANE', scale=[10, 10, 1], location=[0, 0, -0.25], rotation=[0, 0, 0]),
        bproc.object.create_primitive('PLANE', scale=[10, 10, 1], location=[0, 10, 0], rotation=[-1.570796, 0, 0]),    
        bproc.object.create_primitive('PLANE', scale=[10, 10, 1], location=[0, -10, 0], rotation=[1.570796, 0, 0]),     
        bproc.object.create_primitive('PLANE', scale=[10, 10, 1], location=[10, 0, 0], rotation=[0, 1.570796, 0]),      
        bproc.object.create_primitive('PLANE', scale=[10, 10, 1], location=[-10, 0, 0], rotation=[0, -1.570796, 0]),    
    ]
    for plane in planes:
        plane.enable_rigidbody(False, collision_shape='BOX', mass=1.0, friction = 100.0, linear_damping = 0.99, angular_damping = 0.99)
    
    return planes

def add_lights():
    light_plane = bproc.object.create_primitive('PLANE', scale=[10, 10, 1], location=[0, 0, 10])
    light_plane.set_name('light_plane')
    light_plane_material = bproc.material.create('light_material')

    light_points = [bproc.types.Light() for _ in range(4)]
    for point in light_points:
        point.set_energy(200)

    light_points[0].set_location([0, 5, 5])
    light_points[1].set_location([0, -5, 5])
    light_points[2].set_location([5, 0, 5])
    light_points[3].set_location([-5, 0, 5])

    return light_plane, light_plane_material, light_points

def set_random_intrinsics(cfg):
    focal_length = np.random.uniform(*cfg.cam.focal_range)
    diff_focal = np.random.uniform(*cfg.cam.diff_focal_range) # [1-x,1+x]

    pixel_aspect_x = pixel_aspect_y = 1
    if diff_focal >= 1:
        pixel_aspect_y = diff_focal
    else:
        pixel_aspect_x = 2 - diff_focal

    bproc.camera.set_intrinsics_from_blender_params(
        lens=focal_length, pixel_aspect_x=pixel_aspect_x,
        pixel_aspect_y=pixel_aspect_y, lens_unit="MILLIMETERS")
        
    return focal_length

def set_random_extrinsics(cfg, focal_length):
    cam_dist_base = cfg.cam.dist_base
    cam_dist_delta = cfg.cam.dist_delta
    elevation_min, elevation_max = cfg.cam.elev_range

    # Adjust camera distance based on focal range
    focal_base = cfg.cam.focal_base
    cam_dist_base *= focal_length / focal_base
    radius_min, radius_max = cam_dist_base - cam_dist_delta, cam_dist_base + cam_dist_delta

    locs = []
    for frame_id in range(cfg.cam.num_views):
        location = bproc.sampler.shell(
            center=[0, 0, 0],
            radius_min=radius_min, radius_max=radius_max,
            elevation_min=elevation_min, elevation_max=elevation_max)
        
        # Randomly sample a lookAt point close to the object center and a plane rotation
        poi = bproc.sampler.shell(center=[0, 0, 0], radius_min=0, radius_max=cfg.cam.poi_dist_max)
        max_plane_rot = cfg.cam.max_rot * np.pi / 180
        inplane_rot = np.random.uniform(-max_plane_rot, max_plane_rot)
    
        # Compute rotation based on vector going from location towards poi
        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=inplane_rot)
        cam2world = bproc.math.build_transformation_mat(location, rotation_matrix)

        # Replace previous camera by forcing the frame_id
        bproc.camera.add_camera_pose(cam2world, frame=frame_id)
        locs.append(cam2world)

def set_random_lighting(scene, cfg):
    # If using HDRI, just set the background
    if cfg.use_hdri:
        hdri_path = np.random.choice(scene.hdri_files)
        logging.info(f"Using HDRI from {hdri_path}")
        bproc.world.set_world_background_hdr_img(hdri_path)
        return

    # Randomize light plane color and emissivity
    scene.light_plane_material.make_emissive(
        emission_strength=np.random.uniform(3,6), 
        emission_color=np.random.uniform([0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0]))  
    scene.light_plane.replace_materials(scene.light_plane_material)

    # Randomize light point color and pose
    for point in scene.light_points:
        point.set_color(np.random.uniform([0.5,0.5,0.5],[1,1,1]))

def compute_pcd(data, cfg):
    if not cfg.generate_pcd:
        return trimesh.PointCloud(vertices=np.array([]))

    pc_all = []
    for i in range(cfg.cam.num_views, cfg.cam.num_views + cfg.pcd.num_views):
        # Point coordinates
        pc_xyz = bproc.camera.pointcloud_from_depth(data["depth"][i], frame=i) # [H, W, 3]
        pc_xyz = pc_xyz[data["depth"][i] < 100] # [N, 3]

        # Point color values
        pc_rgb = data["colors"][i] # [H, W, 3]
        pc_rgb = pc_rgb[data["depth"][i] < 100] # [N, 3]
    
        pc = np.concatenate([pc_xyz, pc_rgb], axis=-1) # [N, 6]
        pc_all.append(pc)

    # Concatenate all point clouds and randomly subsample
    points = np.concatenate(pc_all, axis=0)
    points = points[np.random.choice(points.shape[0], cfg.pcd.num_points, replace=False)]
    return trimesh.PointCloud(vertices=points[:, :3], colors=points[:, 3:])

def render_object(data_path, writer, output_path, objaverse_id, scene, cfg):
    obj = scene.objects[objaverse_id]
    obj.hide(False)

    set_random_lighting(scene, cfg)
    focal_length = set_random_intrinsics(cfg)
    set_random_extrinsics(cfg, focal_length)

    data = bproc.renderer.render()
    logging.info(f"Finished rendering object: {objaverse_id}. Saving data to disk.")

    data["intr"] = bproc.camera.get_intrinsics_as_K_matrix()
    data["extr"] = [bproc.camera.get_camera_pose(f) for f in range(cfg.cam.num_views)]
    data["pcd"] = compute_pcd(data, cfg)
    data["num_views"] = cfg.cam.num_views
    writer.write(objaverse_id, data)
    
    # DEBUG STUFF
    for i in range(cfg.cam.num_views):
        # RGB
        image = data["colors"][i]
        image_path = os.path.join(output_path, f"color_{i:04d}.png")
        cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA))

        # Depth
        depth = data["depth"][i]
        depth[depth > 65535] = 0
        # depth = np.round(depth).astype(np.uint16)

        # depth_path = os.path.join(output_path, f"depth_{i:04d}.png")
        # img = Image.fromarray(depth)
        # img.save(depth_path)

        depth_min = depth.min()
        depth_max = depth.max()
        depth_normalized = (depth - depth_min) / (depth_max - depth_min)

        # Apply a colormap using Matplotlib
        colormap='viridis'
        colormap = plt.get_cmap(colormap)
        depth_colored = colormap(depth_normalized)  # Returns an RGBA image

        # Convert to 8-bit RGB and save as PNG
        depth_colored = (depth_colored[:, :, :3] * 255).astype(np.uint8)

        # Save the colored depth image
        depth_path = os.path.join(output_path, f"depth_{i:04d}.png")
        plt.imsave(depth_path, depth_colored)

        # Save the center-cropped image
        cropped_image = center_crop(image, 224)
        cropped_image_path = os.path.join(output_path, f"{i:04d}_cropped.png")
        cv2.imwrite(cropped_image_path, cv2.cvtColor(cropped_image, cv2.COLOR_RGBA2BGRA))

    obj.hide(True)

def center_crop(image, crop_size):
    height, width = image.shape[:2]
    new_size = crop_size

    # Calculate the coordinates for the crop
    start_x = (width - new_size) // 2
    start_y = (height - new_size) // 2

    # Perform the cropping
    cropped_image = image[start_y:start_y + new_size, start_x:start_x + new_size]

    return cropped_image

def main():
    parser = argparse.ArgumentParser(description="Render a batch of Objaverse assets.")
    parser.add_argument("--config", type=str, help="Path to config file", default="config/config.json")
    parser.add_argument("--data_path", type=str, help="Path to the Objaverse dataset.")
    parser.add_argument("--start_idx", type=int, help="Index of the first object to render in 'objaverse_models.json'.")
    parser.add_argument("--num_objects", type=int, help="Number of objects to render.")
    parser.add_argument("--output_dir", type=str, help="Path to save the rendered models.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for data randomization. Default is random.")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    with open(args.config, "r") as f:
        cfg = json.load(f, object_hook=lambda x: SimpleNamespace(**x))
    
    with open(os.path.join(args.data_path, "objaverse_models.json"), "r") as f:
        obj_list = json.load(f)
    
    # Each renderer writes to its own log file and output file
    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, f"{args.start_idx:06d}.log"),
        level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    writer = TarWriter(args.output_dir, args.config, args.start_idx)

    # Initialize all static scene components
    objects = obj_list[args.start_idx : args.start_idx + args.num_objects]
    scene = setup_blender(args.data_path, objects, cfg)
    logging.info(f"Finished setting up static scene.")

    for obj in objects:
        obj_id = obj["objaverse_id"]
        output_path = os.path.join(args.output_dir, obj_id)
        shutil.rmtree(output_path, ignore_errors=True)
        os.makedirs(output_path)
    
        logging.info(f"Rendering Object {obj_id} to {os.path.abspath(output_path)}")
        render_object(args.data_path, writer, output_path, obj_id, scene, cfg)

# blenderproc run batch_renderer.py
if __name__ == "__main__":
    main()