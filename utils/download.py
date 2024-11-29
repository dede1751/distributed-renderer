"""
Modified version of the objaverse_json.py script from: https://github.com/simonschlaepfer/BlenderProcObjaverse
Downloads and processes Objaverse meshes in parallel:
 - Converts '.glb' to '.obj'
 - Normalizes mesh to fit into a unit cube
 - Centers the mesh into the origin
"""

import os # Set env vars before importing libraries
os.environ["OPENBLAS_NUM_THREADS"] = "1" 
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import json
from typing import Optional
import argparse
import random
import shutil
import multiprocessing

import numpy as np
import objaverse
from tqdm import tqdm
import trimesh 
from pygltflib import GLTF2


def load_existing_json(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)
    return []


def save_to_json(file_path, data):
    directory = os.path.dirname(file_path)
    os.makedirs(directory, exist_ok=True)
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


def convert_glb_to_obj(glb_path: str, obj_path: str):
    # Load the mesh or collection of meshes
    try:
        trimesh_mesh = trimesh.load(glb_path)
    except:
        raise Exception(f"Mesh could not be imported. Skipped {glb_path}")

    # Check if the loaded object is a scene (which can contain multiple geometries)
    if isinstance(trimesh_mesh, trimesh.Scene):
        for name, geom in trimesh_mesh.geometry.items():
            if not isinstance(geom, trimesh.Trimesh):
                raise Exception(f"Mesh contains unsupported Geometry. Skipped {glb_path}")

        merged_mesh = trimesh.util.concatenate(trimesh_mesh.dump())
    else:
        # If trimesh_mesh is already a single mesh, no need to concatenate
        merged_mesh = trimesh_mesh
    
    # Normalize scale to fit in a unit bounding box
    norm_scale = max(merged_mesh.bounding_box.extents)
    if norm_scale > 0:
        merged_mesh.apply_scale(1 / norm_scale)

    # Center the object on the mesh centroid
    centroid = merged_mesh.centroid
    merged_mesh.apply_translation(-centroid)

    merged_mesh.export(obj_path, include_texture=True, file_type='obj')
    return norm_scale


def process_object(obj_details):
    objaverse_base_path, glb_path, objaverse_id = obj_details
    github_id = glb_path.split('/')[-2]
    obj_path = os.path.join(objaverse_base_path, "objs", github_id, objaverse_id, 'model_scaled.obj')
    os.makedirs(os.path.dirname(obj_path), exist_ok=True)

    try:
        norm_scale = convert_glb_to_obj(glb_path, obj_path)
    except Exception as e:
        return False, e

    return True, {
        "objaverse_id": objaverse_id,
        "github_id": github_id,
        "norm_scale": norm_scale
    }


def update_glb_json(objaverse_dict, objaverse_base_path, json_path, filter_skins):
    existing_data = load_existing_json(json_path)
    existing_map = {item["objaverse_id"] for item in existing_data}

    skinned = 0
    for objaverse_id, glb_path in tqdm(objaverse_dict.items(), desc="Copying cached files."):
        if objaverse_id in existing_map:
            continue

        # 1817 skinned meshes out of 42495 total
        if filter_skins:
            gltf = GLTF2().load(glb_path)
            if gltf.skins:
                del objaverse_dict[objaverse_id]
                skinned += 1
                continue

        github_id = glb_path.split('/')[-2]
        dst_dir = os.path.join(objaverse_base_path, "glbs", github_id)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copyfile(glb_path, os.path.join(dst_dir, f"{objaverse_id}.glb"))

        existing_data.append({
            "objaverse_id": objaverse_id,
            "github_id": github_id,
        })

    save_to_json(json_path, existing_data)
    print(f"Skipped a total of {skinned} skinned objects.")


def update_obj_json(objaverse_dict, objaverse_base_path, json_path, num_workers=32):
    existing_data = load_existing_json(json_path)
    existing_map = {item["objaverse_id"] for item in existing_data}

    obj_details = [
        (objaverse_base_path, glb_path, objaverse_id)
        for objaverse_id, glb_path in objaverse_dict.items()
        if objaverse_id not in existing_map
    ]
    print(f"Processing {len(obj_details)} objects with {num_workers} workers...")

    with multiprocessing.Pool(num_workers) as pool:
        results = tqdm(pool.imap_unordered(process_object, obj_details), total=len(obj_details))
        errors = []

        for idx, (success, data) in enumerate(results):
            if success:
                existing_data.append(data)
                if idx % 250 == 0:
                    save_to_json(json_path, existing_data)
            else:
                errors.append(data)

        save_to_json(json_path, existing_data)
    
    for error in errors:
        print(error)
    print(f"Skipped a total of {len(errors)} objects.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Objaverse objects and convert them to '.obj' format.")
    parser.add_argument('--data_path', type=str, help="Path to folder in which the dataset will be saved.", required=True)
    parser.add_argument('--num_objects', type=int, help="Number of objects to download. Deafault is the full list.", default=None)
    parser.add_argument('--num_workers', type=int, help="Number of download processes to instantiate.", default=32)
    parser.add_argument('--list_file', type=str, help="Path to a list of UIDs to download. Default is random UIDs.", default=None)
    parser.add_argument('--filter_skins', type=bool, help="Whether to filter skinned objects. Default is False.", default=False)
    args = parser.parse_args()

    objaverse_base_path = os.path.join(args.data_path, "objaverse")
    glb_json_path = os.path.join(objaverse_base_path, "glb_list.json")
    obj_json_path = os.path.join(objaverse_base_path, "obj_list.json")
    os.makedirs(objaverse_base_path, exist_ok=True)

    seed = 42
    random.seed(seed)
    np.random.seed(seed)

    if args.list_file is None:
        print("Loading random objects")
        uids = objaverse.load_uids()
    else:
        print(f"Loading objects from: {args.list_file}")
        with open(args.list_file, "r") as f:
            uids = [uid.strip() for uid in f.readlines()]
    
    if args.num_objects is not None:
        num_objects = min(args.num_objects, len(uids))
        uids = random.sample(uids, num_objects)

    objaverse_dict = objaverse.load_objects(uids=uids, download_processes=args.num_workers)

    print(f"Objects loaded, copying '.glb' files to {objaverse_base_path}/glbs")
    update_glb_json(objaverse_dict, objaverse_base_path, glb_json_path, args.filter_skins)
    print("Converting '.glb' to '.obj'")
    update_obj_json(objaverse_dict, objaverse_base_path, obj_json_path, num_workers=args.num_workers)
    print(f"Saved '.obj' files to {objaverse_base_path}/objs")
