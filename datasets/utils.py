import argparse
import os
import json

from tqdm import tqdm

def collect_glb_files(root_folder):
    """
    Collect all '.glb' files from a root folder. Assumes files are only in subdirectories of root.
    """
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
    """
    Save a python object to a JSON file, creating the necessary directories if they don't exist.
    """
    directory = os.path.dirname(file_path)
    os.makedirs(directory, exist_ok=True)
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


def deduplicate_json(src_file, tgt_file, out_file):
    """
    Deduplicate a JSON file based on the "uid" field.
    Removes entries from the target file that are present in the source file.
    """
    with open(src_file, "r") as src_f:
        src_list = json.load(src_f)

    with open(tgt_file, "r") as tgt_f:
        tgt_list = json.load(tgt_f)


    print(f"Loaded {len(src_list)} entries from {src_file} and {len(tgt_list)} entries from {tgt_file}.")
    src_dict = {entry["uid"]: entry for entry in src_list}
    tgt_dedup = [entry for entry in tgt_list if entry["uid"] not in src_dict]

    save_to_json(out_file, tgt_dedup)
    print(f"Deduplication complete, removed {len(tgt_list) - len(tgt_dedup)} duplicates. Output written to {out_file}.")


if __name__ == """__main__""":
    parser = argparse.ArgumentParser(description="Utility tools for the renderer datasets.")
    subparsers = parser.add_subparsers(dest="command")

    dedup_parser = subparsers.add_parser("dedup", help="Deduplicate JSON files based on UID. Removes entries from the target file that are present in the source file.")
    dedup_parser.add_argument("--src", required=True, help="Path the the JSON file to look for duplicates in.")
    dedup_parser.add_argument("--tgt", required=True, help="Path the the JSON file to remove duplicates from.")
    dedup_parser.add_argument("--out", required=True, help="Path for the output JSON file.")

    args = parser.parse_args()

    if args.command == "dedup":
        deduplicate_json(args.src, args.tgt, args.out)
    else:
        parser.print_help()
