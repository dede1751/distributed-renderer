"""
Create a JSON file for the Objaverse dataset, meant to be used with the renderer.
We download the objects when needed and permanently cache them in the data_path folder.
"""

import os # Set env vars before importing libraries
os.environ["OPENBLAS_NUM_THREADS"] = "1" 
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import argparse
import random
import shutil

import numpy as np
import objaverse
from tqdm import tqdm

from utils import collect_glb_files, save_to_json


# Cache to a temporary directory due to cluster limits.
objaverse.BASE_PATH = os.path.join(os.environ["TMPDIR"], ".objaverse")
objaverse._VERSIONED_PATH = os.path.join(objaverse.BASE_PATH, "hf-objaverse-v1")


def cache_glb_files(root_folder, objaverse_dict):
    glb_files_dict = {}

    for objaverse_id, glb_path in tqdm(objaverse_dict.items()):
        github_id = glb_path.split('/')[-2]
        cache_dir = os.path.join(root_folder, github_id)
        os.makedirs(cache_dir, exist_ok=True)
    
        new_path = os.path.join(cache_dir, f"{objaverse_id}.glb")
        shutil.move(glb_path, os.path.join(cache_dir, new_path))
        glb_files_dict[objaverse_id] = new_path

    return glb_files_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Objaverse objects and setup a JSON list.")
    parser.add_argument('--data_path', type=str, help="Path to folder in which the dataset will be saved.", required=True)
    parser.add_argument('--list_file', type=str, help="Path to a list of UIDs to download.  Default is all Objaverse objects.", default=None)
    parser.add_argument('--json_file', type=str, help="Path for the JSON output list.", required=True)
    parser.add_argument('--num_objects', type=int, help="Number of objects to download. Deafault is the full list.", default=None)
    parser.add_argument('--num_workers', type=int, help="Number of download processes to instantiate.", default=32)
    args = parser.parse_args()

    print(f"Fetching all '.glb' files from: {args.data_path}")
    data_path = os.path.abspath(args.data_path)
    os.makedirs(data_path, exist_ok=True)

    seed = 42
    random.seed(seed)
    np.random.seed(seed)

    cached_uid_paths = collect_glb_files(data_path)
    print(f"Found {len(cached_uid_paths)} '.glb' files.")

    if args.list_file is not None:
        print(f"Using only objects from: {args.list_file}")
        with open(args.list_file, "r") as f:
            uid_list = [uid.strip() for uid in f.readlines()]
            cached_uids = {uid: cached_uid_paths[uid] for uid in uid_list if uid in cached_uid_paths}
    else:
        print("Using random objects.")
        uid_list = objaverse.load_uids()
        cached_uids = cached_uid_paths
    
    if args.num_objects is not None:
        num_objects = min(args.num_objects, len(uid_list))
        print(f"Selecting {num_objects} objects...")
        uid_list = random.sample(uid_list, num_objects)
    
    dl_uids = [uid for uid in uid_list if uid not in cached_uids]
    if (len(dl_uids) > 0):
        print(f"Fetching {len(dl_uids)} new objects...")
        objaverse_dict = objaverse.load_objects(uids=dl_uids, download_processes=args.num_workers)

        print(f"Moving {len(objaverse_dict)} new '.glb' files to {data_path}...")
        new_uids = cache_glb_files(data_path, objaverse_dict)
        
        print(f"Merging cached and new '.glb' files...")
        cached_uids.update(new_uids)
    else:
        print(f"All files are already cached!")

    print(f"Saving {len(cached_uids)} objects to: {args.json_file}")
    uid_json = [
        {
            "uid": uid,
            "cat_id": path.split('/')[-2], # here cat_id is github_id
            "glb_path": path,
        } for uid, path in cached_uids.items()
    ]
    save_to_json(args.json_file, uid_json)
