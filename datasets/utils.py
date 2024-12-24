import os
import json
import tarfile

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


def resume_render(output_dir, old_file, new_file):
    """
    Generate a new JSON list to resume an incomplete rendering job.
    """
    # Collect all the UIDs that have been rendered so far, as well as the maximum shard index.
    max_idx = 0
    rendered_uids = set()
    for shard_file in os.listdir(os.path.join(output_dir, "shards")):
        print(f"Processing {shard_file}...")
        shard_idx = int(os.path.splitext(shard_file)[0].split("_")[1])
        max_idx = max(max_idx, shard_idx)

        tar_file = os.path.join(output_dir, "shards", shard_file)
        with tarfile.open(tar_file, "r") as tar:
            members = [member.name for member in tar.getmembers() if member.isdir()]
            rendered = {member.split("/")[0] for member in members}
            rendered_uids.update(rendered)
            print(f"Found {len(rendered)} rendered objects.")
    print(f"Found {len(rendered_uids)} rendered entries. Maximum shard index is {max_idx}.")
    
    # Remove rendered UIDs from the original list.
    with open(old_file, "r") as old_f:
        old_list = json.load(old_f)
    new_list = [entry for entry in old_list if entry["uid"] not in rendered_uids]

    save_to_json(new_file, new_list)
    print(f"Generated resume list with {len(new_list)} entries. Output written to {new_file}.")
