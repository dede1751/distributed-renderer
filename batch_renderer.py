import blenderproc as bproc
from blenderproc.python.camera import CameraUtility

import urllib
import os
import json
import math
import argparse
from typing import List, Tuple, Dict
import shutil
from dataclasses import dataclass
import logging

import bpy
import numpy as np
import cv2
import trimesh


@dataclass
class BlenderScene:
    """Keep track of all elements in the scene."""
    objects: Dict[str, bproc.types.MeshObject] # objaverse_id -> MeshObject
    planes: List[bproc.types.MeshObject]
    light_plane: bproc.types.MeshObject
    light_plane_material: bproc.types.Material
    light_point: bproc.types.MeshObject

def setup_blender(data_path, objects, cfg):
    # Initialize Blender with EEVEE engine
    bproc.init()
    bpy.context.scene.render.engine = "BLENDER_EEVEE_NEXT"

    # Initialize fixed components of the scene and load all objects
    loaded_objects = load_objects(data_path, objects)
    planes = add_bowl()
    light_plane, light_plane_material, light_point = add_light()
    scene = BlenderScene(loaded_objects, planes, light_plane, light_plane_material, light_point)

    # Initialize renderer settings
    bproc.renderer.enable_depth_output(activate_antialiasing=False, convert_to_distance=False)
    if cfg["output_normals"]:
        bproc.renderer.enable_normals_output()
    if cfg["output_diffuse"]:
        bproc.renderer.enable_diffuse_color_output()

    #bproc.renderer.set_output_format(enable_transparency=True)
    bproc.renderer.set_max_amount_of_samples(50)
    bproc.renderer.set_cpu_threads(32)

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

def add_bowl():
    """Add rectangular bowl to the scene."""
    planes = [
        bproc.object.create_primitive('PLANE', scale=[4, 4, 0.001], location=[0, 0, -1], rotation=[0, 0, 0]),
        bproc.object.create_primitive('PLANE', scale=[4, 4, 0.001], location=[0, 4, 0], rotation=[-1.570796, 0, 0]),    
        bproc.object.create_primitive('PLANE', scale=[4, 4, 0.001], location=[0, -4, 0], rotation=[1.570796, 0, 0]),     
        bproc.object.create_primitive('PLANE', scale=[4, 4, 0.001], location=[4, 0, 0], rotation=[0, 1.570796, 0]),      
        bproc.object.create_primitive('PLANE', scale=[4, 4, 1], location=[-4, 0, 0], rotation=[0, -1.570796, 0]),    
    ]
    for plane in planes:
        plane.enable_rigidbody(False, collision_shape='BOX', mass=1.0, friction = 100.0, linear_damping = 0.99, angular_damping = 0.99)
    
    return planes

def add_light():
    light_plane = bproc.object.create_primitive('PLANE', scale=[4, 4, 1], location=[0, 0, 8])
    light_plane.set_name('light_plane')
    light_plane_material = bproc.material.create('light_material')

    light_point = bproc.types.Light()
    light_point.set_energy(200)

    return light_plane, light_plane_material, light_point

def set_random_intrinsics(cfg):
    focal_length = np.random.uniform(*cfg["focal_range"])
    diff_focal = np.random.uniform(*cfg["diff_focal_range"]) # [1-x,1+x]

    pixel_aspect_x = pixel_aspect_y = 1
    if diff_focal >= 1:
        pixel_aspect_y = diff_focal
    else:
        pixel_aspect_x = 2 - diff_focal

    bproc.camera.set_intrinsics_from_blender_params(
        lens=focal_length, pixel_aspect_x=pixel_aspect_x,
        pixel_aspect_y=pixel_aspect_y, lens_unit="MILLIMETERS")

def set_random_extrinsics(cfg):
    radius_min, radius_max = cfg["cam_dist_range"]
    elevation_min, elevation_max = cfg["cam_elev_range"]

    locs = []
    for frame_id in range(cfg["num_views"]):
        location = bproc.sampler.shell(
            center=[0, 0, 0],
            radius_min=radius_min, radius_max=radius_max,
            elevation_min=elevation_min, elevation_max=elevation_max)
        print(f"{frame_id}, {location}, {np.linalg.norm(location)}")
        
        # Randomly sample a lookAt point close to the object center and a plane rotation
        poi = bproc.sampler.shell(center=[0, 0, 0], radius_min=0, radius_max=cfg["poi_dist_max"])
        max_plane_rot = cfg["cam_max_rot"] * np.pi / 180
        inplane_rot = np.random.uniform(-max_plane_rot, max_plane_rot)
    
        # Compute rotation based on vector going from location towards poi
        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=inplane_rot)
        cam2world = bproc.math.build_transformation_mat(location, rotation_matrix)

        # Replace previous camera by forcing the frame_id
        bproc.camera.add_camera_pose(cam2world, frame=frame_id)
        locs.append(cam2world)

def set_random_lighting(scene, cfg):
    # Randomize light plane color and emissivity
    scene.light_plane_material.make_emissive(
        emission_strength=np.random.uniform(3,6), 
        emission_color=np.random.uniform([0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0]))  
    scene.light_plane.replace_materials(scene.light_plane_material)

    # Randomize light point color and pose
    scene.light_point.set_color(np.random.uniform([0.5,0.5,0.5],[1,1,1]))
    location = bproc.sampler.shell(
        center = [0, 0, 0],
        radius_min = 3.5, radius_max = 3.9,
        elevation_min = 5, elevation_max = 89)
    scene.light_point.set_location(location)

def render_object(data_path, output_path, objaverse_id, scene, cfg):
    obj = scene.objects[objaverse_id]
    obj.hide(False)

    set_random_lighting(scene, cfg)
    set_random_intrinsics(cfg)
    set_random_extrinsics(cfg)

    data = bproc.renderer.render()
    for i in range(cfg["num_views"]):
        # RGB
        image = data["colors"][i]
        image_path = os.path.join(output_path, f"{i:04d}.png")
        cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA))

        # Depth
        depth_path = os.path.join(output_path, f"{i:04d}.npy")
        np.save(depth_path, data["depth"][i])


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
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = json.load(f)
    
    with open(os.path.join(args.data_path, "objaverse_models.json"), "r") as f:
        obj_list = json.load(f)
    
    # Each renderer writes to its own log file
    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, f"{args.start_idx}.log"),
        level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
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
        render_object(args.data_path, output_path, obj_id, scene, cfg)

# blenderproc run batch_renderer.py
if __name__ == "__main__":
    main()