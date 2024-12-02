import blenderproc as bproc
from blenderproc.python.types.MeshObjectUtility import create_with_empty_mesh

import json
import argparse
from typing import List, Tuple, Dict
from types import SimpleNamespace
import logging
import random

import bpy
import numpy as np

# Blenderproc sets PYTHONPATH independently
import os
import sys
sys.path.insert(0, os.path.abspath("./"))

from utils.writer import TarWriter


def normalize_obj(obj: bproc.types.MeshObject):
    """Normalize and center a MeshObject within the unit cube."""
    bbox = obj.get_bound_box()
    min_corner, max_corner = bbox[0], bbox[6]

    center = (min_corner + max_corner) / 2
    length = np.max(max_corner - min_corner)

    if length > 0:
        scale = 1 / length
    else:
        logging.error(f"Scale factor for {obj.blender_obj.name} is 0.")
        scale = 1

    obj.set_location(obj.get_location() - center)
    obj.persist_transformation_into_mesh()
    obj.set_scale(obj.get_scale() * scale)
    obj.persist_transformation_into_mesh() 


class BlenderScene:
    """Keep track of all elements in the scene."""

    def __init__(self, data_path: str, writer: TarWriter, cfg: SimpleNamespace):
        self.cfg = cfg
        self.writer = writer
        self.data_path = data_path
        self.load_hdri()

        bproc.init()
        if cfg.generate_pcd:
            # Add uniformly distributed cameras on a sphere
            for i in range(cfg.pcd.num_views):
                location = bproc.sampler.sphere(center=[0, 0, 0], radius=cfg.pcd.cam_dist, mode='SURFACE')
                rotation_matrix = bproc.camera.rotation_from_forward_vec(-location)
                cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
                bproc.camera.add_camera_pose(cam2world_matrix, frame=cfg.cam.num_views + i)

        # Initialize renderer settings
        bproc.camera.set_resolution(*cfg.resolution)
        bproc.renderer.enable_depth_output(activate_antialiasing=False)
        bproc.renderer.set_output_format(enable_transparency=True)
        #bproc.renderer.set_cpu_threads(2)
    

    def load_hdri(self):
        self.hdri = []

        for root, dirs, files in os.walk(self.cfg.hdri_path):
            for file in files:
                if file.endswith('.exr') or file.endswith('.hdr'):
                    self.hdri.append(os.path.join(root, file))
    
    
    def load_object(self, obj: Dict[str, str]) -> bproc.types.MeshObject:
        if self.cfg.load_glb:
            obj_path = os.path.join(
                self.data_path, "objs", obj["github_id"], obj["objaverse_id"], "model_scaled.obj")

            # OBJs are assumed to be merged and normalized beforehand
            blender_objs = bproc.loader.load_obj(obj_path)
            if len(blender_objs) > 1:
                logging.error(f"Loaded multiple objects at once from: {obj_path}")
            merged_obj = blender_objs[-1]
            
            merged_obj.set_shading_mode('auto')
        else:
            obj_path = os.path.join(
                self.data_path, "glbs", obj["github_id"], f"{obj['objaverse_id']}.glb")
            blender_objs = bproc.loader.load_obj(obj_path)

            # Merge meshes into a single object
            merged_obj = create_with_empty_mesh(obj['objaverse_id'])
            merged_obj.join_with_other_objects(blender_objs)
            normalize_obj(merged_obj)
        
        merged_obj.set_shading_mode('auto')
        return merged_obj


    def set_random_intrinsics(self):
        cfg = self.cfg.cam

        focal_length = np.random.uniform(*self.cfg.cam.focal_range)
        diff_focal = np.random.uniform(*self.cfg.cam.diff_focal_range) # [1-x,1+x]

        pixel_aspect_x = pixel_aspect_y = 1
        if diff_focal >= 1:
            pixel_aspect_y = diff_focal
        else:
            pixel_aspect_x = 2 - diff_focal

        bproc.camera.set_intrinsics_from_blender_params(
            lens=focal_length, pixel_aspect_x=pixel_aspect_x,
            pixel_aspect_y=pixel_aspect_y, lens_unit="MILLIMETERS")
            
        return focal_length


    def set_random_extrinsics(self, focal_length):
        cfg = self.cfg.cam
    
        cam_dist_base = cfg.dist_base
        cam_dist_delta = cfg.dist_delta
        elevation_min, elevation_max = cfg.elev_range

        # Adjust camera distance based on focal range
        focal_base = cfg.focal_base
        cam_dist_base *= focal_length / focal_base
        radius_min, radius_max = cam_dist_base - cam_dist_delta, cam_dist_base + cam_dist_delta

        locs = []
        for frame_id in range(cfg.num_views):
            location = bproc.sampler.shell(
                center=[0, 0, 0],
                radius_min=radius_min, radius_max=radius_max,
                elevation_min=elevation_min, elevation_max=elevation_max)
            
            # Randomly sample a lookAt point close to the object center and a plane rotation
            poi = bproc.sampler.shell(center=[0, 0, 0], radius_min=0, radius_max=cfg.poi_dist_max)
            max_plane_rot = cfg.max_rot * np.pi / 180
            inplane_rot = np.random.uniform(-max_plane_rot, max_plane_rot)
        
            # Compute rotation based on vector going from location towards poi
            rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=inplane_rot)
            cam2world = bproc.math.build_transformation_mat(location, rotation_matrix)

            # Replace previous camera by forcing the frame_id
            bproc.camera.add_camera_pose(cam2world, frame=frame_id)
            locs.append(cam2world)


    def set_random_hdri(self):
        hdri_path = np.random.choice(self.hdri)
        logging.info(f"Using HDRI from {hdri_path}")
        bproc.world.set_world_background_hdr_img(hdri_path)


    def toggle_pcd_rendering(self, value: bool):
        cfg = self.cfg

        if value:
            bpy.context.scene.frame_start = cfg.cam.num_views
            bpy.context.scene.frame_end = cfg.cam.num_views + cfg.pcd.num_views
            bproc.renderer.set_max_amount_of_samples(1)
        else:
            bpy.context.scene.frame_start = 0
            bpy.context.scene.frame_end = cfg.cam.num_views
            bproc.renderer.set_max_amount_of_samples(cfg.num_samples)


    def compute_pcd(self, depths):
        pc_all = []
        for i, depth in enumerate(depths):
            pc_xyz = bproc.camera.pointcloud_from_depth(
                depth,
                frame=self.cfg.cam.num_views + i,
                depth_cut_off=65536) # [H, W, 3]
            pc_xyz = pc_xyz[depth < 100].reshape([-1, 3]) # [N, 3]
            pc_all.append(pc_xyz)

        # Concatenate all point clouds and randomly subsample
        points = np.concatenate(pc_all, axis=0)
        logging.info(f"Generated PointCloud with {points.shape[0]} points.")

        if points.shape[0] < self.cfg.pcd.num_points:
            logging.error(f"PointCloud has less than {self.cfg.pcd.num_points} points!")
        else:
            points = points[np.random.choice(points.shape[0], self.cfg.pcd.num_points, replace=False)]

        return points


    def render_object(self, object):
        objaverse_id = object["objaverse_id"]
        logging.info(f"Started rendering Object {objaverse_id}.")
        obj = self.load_object(object)

        self.set_random_hdri()
        focal_length = self.set_random_intrinsics()
        self.set_random_extrinsics(focal_length)

        self.toggle_pcd_rendering(False)
        data = bproc.renderer.render()
        logging.info(f"Finished rendering views of object: {objaverse_id}.")

        if self.cfg.generate_pcd:
            self.toggle_pcd_rendering(True)
            pcd_data = bproc.renderer.render()
            logging.info(f"Finished rendering pcd views of object: {objaverse_id}.")
            data["pcd"] = self.compute_pcd(pcd_data["depth"])
        else:
            data["pcd"] = np.array([])

        data["intr"] = bproc.camera.get_intrinsics_as_K_matrix()
        data["extr"] = [bproc.camera.get_camera_pose(f) for f in range(self.cfg.cam.num_views)]
        data["num_views"] = self.cfg.cam.num_views

        self.writer.write(objaverse_id, data)
        bproc.object.delete_multiple([obj], remove_all_offspring=True)


def main():
    parser = argparse.ArgumentParser(description="Render a batch of Objaverse assets.")
    parser.add_argument("--config", type=str, help="Path to config file", default="config/config.json")
    parser.add_argument("--data_path", type=str, help="Path to the Objaverse dataset.", required=True)
    parser.add_argument("--shard_idx", type=int, help="Index of the dataset shard.", required=True)
    parser.add_argument("--num_workers", type=int, help="Total number of workers to shard the dataset across.", required=True)
    parser.add_argument("--max_objects", type=int, help="Maximum objects to render.", default=None)
    parser.add_argument("--output_dir", type=str, help="Path to save the rendered models.", required=True)
    parser.add_argument("--seed", type=int, help="Seed for data randomization. Default is random.", default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    with open(args.config, "r") as f:
        cfg = json.load(f, object_hook=lambda x: SimpleNamespace(**x))
    
    if cfg.load_glb:
        json_path = os.path.join(args.data_path, "glb_list.json")
    else:
        json_path = os.path.join(args.data_path, "obj_list.json")

    with open(json_path, "r") as f:
        obj_list = json.load(f)
    
    # Each renderer writes to its own log file and output file
    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, f"{args.shard_idx:06d}.log"),
        level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Shard the dataset so the last shard is potentially smaller than the others.
    total_objects = len(obj_list)
    obj_per_shard = total_objects // args.num_workers
    if total_objects % args.num_workers != 0:
        obj_per_shard += 1
    if args.max_objects is not None:
        obj_per_shard = min(args.max_objects, obj_per_shard)

    start_idx = obj_per_shard * args.shard_idx
    end_idx = min(start_idx + obj_per_shard, total_objects)
    objects = obj_list[start_idx: end_idx]
    logging.info(f"Started rendering {len(objects)} objects from index {start_idx}.")

    # Initialize all static scene components
    writer = TarWriter(args.output_dir, args.config, args.shard_idx)
    scene = BlenderScene(args.data_path, writer, cfg)
    logging.info(f"Finished setting up static scene.")

    for object in objects:
        scene.render_object(object)
    logging.info(f"Finished rendering.")


# blenderproc run render.py
if __name__ == "__main__":
    main()