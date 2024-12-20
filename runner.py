"""
Runner script, handles sharding and restarting the rendering script to avoid memory leaks.
"""
import argparse
import json
import subprocess
import sys


if __name__ == "__main__":
    cli_args = sys.argv[1:]

    parser = argparse.ArgumentParser(description="Run a single rendering worker by executing the `render.py` script multiple times.")
    parser.add_argument("--json_file", type=str, help="Path to JSON dataset.", required=True)
    parser.add_argument("--shard_idx", type=int, help="Index of the dataset shard.", required=True)
    parser.add_argument("--shard_offset", type=int, help="Shard index offset to avoid overwriting previous runs.", default=0)
    parser.add_argument("--num_workers", type=int, help="Total number of workers to shard the dataset across.", required=True)
    parser.add_argument("--max_objects", type=int, help="Maximum objects to render in the shard.", default=None)
    parser.add_argument("--obj_per_rerun", type=int, help="Maximum number of objects to render before restarting the script. Default is no restart.", default=None)
    
    # Only used by `render.py`.
    parser.add_argument("--output_dir", type=str, help="Path to save the rendered models.", required=True)
    parser.add_argument("--log_resources", action='store_true', help="Log resource (CPU/Mem) usage.", default=False)
    parser.add_argument("--seed", type=int, help="Seed for data randomization. Default is random.", default=None)
    parser.add_argument("--config", type=str, help="Path to config file", default="config/config.json")
    args = parser.parse_args()

    with open(args.json_file, "r") as f:
        obj_list = json.load(f)
    
    # Shard the dataset so the last shard is potentially smaller than the others.
    total_objects = len(obj_list)
    obj_per_shard = total_objects // args.num_workers
    if total_objects % args.num_workers != 0:
        obj_per_shard += 1
    
    # Compute the range of objects to render in this shard, regardless of reruns.
    shard_start = obj_per_shard * args.shard_idx
    shard_end = min(shard_start + obj_per_shard, total_objects)
    if args.max_objects is not None:
        shard_end = min(shard_start + args.max_objects, shard_end)
    
    # Compute the range of objects to render in each rerun.
    if args.obj_per_rerun is None:
        args.obj_per_rerun = total_objects # No reruns, render the full shard in one go.

    runs = []
    for rerun_start in range(shard_start, shard_end, args.obj_per_rerun):
        runs.append((rerun_start, min(rerun_start + args.obj_per_rerun, shard_end)))

    # Modify the original shard_idx argument to add the offset.
    cli_args[cli_args.index("--shard_idx") + 1] = str(args.shard_idx + args.shard_offset)

    # Run the rendering script for each rerun by providing appropriate args.
    for rerun_idx, (rerun_start, rerun_end) in enumerate(runs):
        rerun_args = [
            "--start_idx", str(rerun_start),
            "--end_idx", str(rerun_end),
            "--rerun_idx", str(rerun_idx),
            "--tot_completed", str(rerun_start - shard_start),
            "--tot_objects", str(shard_end - shard_start),
        ]
        cmd = ["blenderproc run render.py"] + cli_args + rerun_args
        subprocess.run(" ".join(cmd), shell=True)
