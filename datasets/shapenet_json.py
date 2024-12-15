"""
Create a '.json' file for the ShapeNetCore-glb dataset, meant to be used with the renderer.
We expect the ShapeNetCore dataset to already be downloaded in the data_path folder.
"""

import os # Set env vars before importing libraries
os.environ["OPENBLAS_NUM_THREADS"] = "1" 
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import json
import argparse
import random

import numpy as np
from tqdm import tqdm


def collect_glb_files(root_folder):
    glb_files_dict = {}

    for dirname in tqdm(os.listdir(root_folder)):
        dir = os.path.join(root_folder, dirname)
        if os.path.isdir(dir):
            for file in os.listdir(dir):
                if file.endswith('.glb'):
                    uid = os.path.splitext(file)[0]
                    glb_files_dict[uid] = os.path.join(dir, file)
                
    return glb_files_dict


def save_to_json(file_path, data):
    directory = os.path.dirname(file_path)
    os.makedirs(directory, exist_ok=True)
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup '.json' list for ShapeNetCore objects (note that you need access to the gated dataset).")
    parser.add_argument('--data_path', type=str, help="Absolute Path to folder in which the dataset is saved.", required=True)
    parser.add_argument('--json_name', type=str, help="Name of the '.json' output list.", required=True)
    parser.add_argument('--list_file', type=str, help="Path to a list of UIDs to use. Default is all ShapeNetCore objects.", default=None)
    parser.add_argument('--num_objects', type=int, help="Maximum number of objects to use. Deafault is the full list.", default=None)
    args = parser.parse_args()

    print(f"Fetching all '.glb' files from: {args.data_path}")
    data_path = os.path.abspath(args.data_path)
    glb_json_path = os.path.join(data_path, f"{args.json_name}.json")

    seed = 42
    random.seed(seed)
    np.random.seed(seed)

    cached_uid_paths = collect_glb_files(data_path)
    print(f"Found {len(cached_uid_paths)} '.glb' files.")

    if args.list_file is not None:
        print(f"Using only objects from: {args.list_file}")
        with open(args.list_file, "r") as f:
            uid_list = [uid.strip() for uid in f.readlines()]
            uids = {uid: cached_uid_paths[uid] for uid in uid_list if uid in cached_uid_paths}

            if len(uids) < len(uid_list):
                print(f"Could not find {len(uid_list) - len(uids)} objects.")
    else:
        print("Using all objects.")
        uids = cached_uid_paths

    uids = uids.items()
    if args.num_objects is not None:
        num_objects = min(args.num_objects, len(uids))
        print(f"Selecting {num_objects} objects...")
        uids = random.sample(uids, num_objects)

    print(f"Saving {len(uids)} objects to: {glb_json_path}")
    uid_json = [
        {
            "uid": uid,
            "cat_id": path.split('/')[-2],
            "glb_path": path,
        } for uid, path in uids
    ]
    save_to_json(glb_json_path, uid_json)