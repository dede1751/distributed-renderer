import os
import yaml
import logging
import multiprocessing
import subprocess
import shutil
from pathlib import Path
from argparse import ArgumentParser

import objaverse

def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def load_object_urls(list_file: str) -> list:
    """Load the URLs for all objects UIDs in the list file. Return a list of tuples (UID, URL)."""
    with open(list_file, "r") as f:
        uids = [uid.strip() for uid in f.readlines()]
    
    # Load UID->URL mapping
    object_paths = objaverse._load_object_paths()
    object_urls = [
        (uid, f"https://huggingface.co/datasets/allenai/objaverse/resolve/main/{object_paths[uid]}")
        for uid in uids
    ]
    return object_urls

def render_object(args):
    (obj_uid, obj_url), output_dir, worker_script, common_args = args

    obj_output_dir = os.path.join(output_dir, obj_uid)
    os.makedirs(obj_output_dir, exist_ok=True)
    
    # Convert common_args to list and add '--' before each key
    common_args_list = []
    for key, value in common_args.items():
        common_args_list.append('--' + str(key))
        common_args_list.append(str(value))

    cmd = [
        "blenderproc", "run", worker_script,
        "--input_url", obj_url,
        "--output", obj_output_dir
    ] + common_args_list

    logging.info(f"Rendering {obj_uid} to {obj_output_dir}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        logging.info(f"Successfully rendered {obj_uid}")
    else:
        logging.error(f"Failed to render {obj_uid}: {result.stderr}")

    return {
        "object": obj_uid,
        "output": obj_output_dir,
        "status": "success" if result.returncode == 0 else "failure",
        "cmd": ' '.join(cmd),
        "error": result.stderr if result.returncode != 0 else None
    }

def main():
    parser = ArgumentParser(description="Distributed rendering of objects using BlenderProc.")
    parser.add_argument('--output_dir', type=str, default='outputs/test', help='Path to the output directory')
    parser.add_argument('--worker_script', type=str, default='worker.py', help='Path to the worker script (worker.py)')
    parser.add_argument('--num_workers', type=int, default=6, help='Number of parallel workers')
    parser.add_argument('--config', type=str, default='config/config.yaml', 
                        help='Path to the YAML file containing common arguments for the worker script')
    parser.add_argument('--list_file', type=str, default='configs/obj_list.txt',
                        help='Path to the .txt file containing the list of UIDs to render')
    args = parser.parse_args()

    shutil.rmtree(args.output_dir, ignore_errors=True)
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(os.path.join(args.output_dir, 'rendering.log'))

    # read the config file and copy it to the output directory
    with open(args.config, 'r') as f:
        common_args = yaml.safe_load(f)
    shutil.copy(args.config, os.path.join(args.output_dir, 'config.yaml'))

    obj_urls = load_object_urls(args.list_file)
    logging.info(f"Found {len(obj_urls)} object files in {args.list_file}")

    pool_args = [(obj, args.output_dir, args.worker_script, common_args) for obj in obj_urls]
    with multiprocessing.Pool(args.num_workers) as pool:
        results = pool.map(render_object, pool_args)

    results_file = os.path.join(args.output_dir, 'results.yaml')
    with open(results_file, 'w') as f:
        yaml.safe_dump(results, f)

    logging.info(f"Rendering completed. Results saved to {results_file}")

if __name__ == "__main__":
    main()