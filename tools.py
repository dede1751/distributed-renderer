"""
Utility tools for the renderer. Supported commands are:
- dedup: Deduplicate JSON files based on UID. Removes entries from the target file that are present in the source file.
- resume: Generate a new JSON list to resume an incomplete rendering job.
- gen_videos: Generate object videos from rendered images.
"""
import argparse
import os
import re

from datasets.utils import deduplicate_json, resume_render
from rendering.videos import generate_all_videos


def find_last_log(lines, pattern):
    for line in lines[::-1]:
        match = pattern.search(line)
        if match:
            return match


def display_progress(log_dir):
    """
    Get a summary of the rendering progress by inspecting the logs.
    For each parallel worker, display:
    - The number of completed objects.
    - The total number of objects to render.
    - The current Rerun index.
    - Time elapsed since the worker started.
    """
    print(f"{'Shard':<10} | {'Rerun':<10} | {'Current':<10} | {'Total':<10}")
    for log_file in os.listdir(log_dir):
        with open(os.path.join(log_dir, log_file), "r") as f:
            lines = f.readlines()

        shard_idx = os.path.splitext(log_file)[0]
        curr_tot = re.compile(r"Rendering object (\d+)/(\d+) --")
        rerun = re.compile(r"RERUN (\d+)")
        current, total = map(int, find_last_log(lines, curr_tot).groups())
        rerun_idx = int(find_last_log(lines, rerun).group(1))
        
        print(f"{shard_idx:<10} | {rerun_idx:<10} | {current:<10} | {total:<10}")
    

if __name__ == """__main__""":
    parser = argparse.ArgumentParser(description="Utility tools for the renderer.")
    subparsers = parser.add_subparsers(dest="command")

    dedup_parser = subparsers.add_parser("dedup", help="Deduplicate JSON files based on UID. Removes entries from the target file that are present in the source file.")
    dedup_parser.add_argument("--src", type=str, required=True, help="Path the JSON file to look for duplicates in.")
    dedup_parser.add_argument("--tgt", type=str, required=True, help="Path the JSON file to remove duplicates from.")
    dedup_parser.add_argument("--out", type=str, required=True, help="Path for output JSON file.")

    resume_parser = subparsers.add_parser("resume", help="Generate a new JSON list to resume an incomplete rendering job.")
    resume_parser.add_argument("--output_dir", type=str, required=True, help="Output directory of the incomplete job.")
    resume_parser.add_argument("--old", type=str, required=True, help="Path the JSON list used for the original rendering job.")
    resume_parser.add_argument("--new", type=str, required=True, help="Path for the new JSON list to resume the rendering job.")

    video_parser = subparsers.add_parser("gen_videos", help="Generate object videos from rendered images.")
    video_parser.add_argument('--in_dir', type=str, required=True, help="Path to uncompressed rendering outputs.")
    video_parser.add_argument('--out_dir', type=str, default="videos", help="Output directory for the videos. Defaults to 'videos'.")
    video_parser.add_argument('--fpi', type=int, default=1, help="Number of frames to display individual views for.")
    video_parser.add_argument('--max_views', type=int, default=None, help="Maximum number of views to load.")

    progress_parser = subparsers.add_parser("progress", help="Display progress of each worker.")
    progress_parser.add_argument("--log_dir", type=str, required=True, help="Path to the directory containing worker logs.")

    args = parser.parse_args()

    if args.command == "dedup":
        deduplicate_json(args.src, args.tgt, args.out)
    elif args.command == "resume":
        resume_render(args.output_dir, args.old, args.new)
    elif args.command == "gen_videos":
        generate_all_videos(args.in_dir, args.out_dir, args.fpi, args.max_views)
    elif args.command == "progress":
        display_progress(args.log_dir)
    else:
        parser.print_help()
