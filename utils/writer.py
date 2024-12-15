"""
Save rendering outputs to an uncompressed Tarball.
"""
import tarfile
import os
import shutil
import logging
from typing import Dict, List, Union

import numpy as np
import cv2

from utils.postprocess import process_view

class TarWriter():
    """
    Save data to a tarball to circumvent cluster file limits.
    Each shard is processed independently and has its own 'output/tmp/xxxxxx' directory, where the
    data is temporarily saved before being added to the shard tarball.
    """
    def __init__(self, output_dir: str, config_path: str, idx: int):
        self.idx = idx

        # Wipe and create new 'tmp' directory, with subdirectories for each data field
        self.tmp_dir = os.path.join(output_dir, "tmp", f"{idx:06d}")
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        os.makedirs(self.tmp_dir, exist_ok=True)

        for cat in ["images", "depth", "extr", "intr"]:
            os.makedirs(os.path.join(self.tmp_dir, cat))

        # Create the shard tarball
        tar_dir = os.path.join(output_dir, "shards")
        os.makedirs(tar_dir, exist_ok=True)

        self.tarfile = os.path.join(tar_dir, f"shard_{idx:06d}.tar")
        logging.info(f"Writing data to {self.tarfile}")

        # Save one copy of config
        config_dest = os.path.join(output_dir, "config.json")
        if not os.path.exists(config_dest):
            shutil.copy(config_path, config_dest)
    
    def write(self, uid: str, data: Dict[str, Union[np.ndarray, List[np.ndarray]]]):
        """
        Postprocess render output and add it to the tarball.
        """
        K = data["intr"]
        for i in range(data["num_views"]):
            image = data["colors"][i][..., :3] # Convert images from RGBA to RGB (we don't care about alpha)
            depth = data["depth"][i]
            extr = data["extr"][i]

            image_path = os.path.join(self.tmp_dir, "images", f"{i:04d}.png")
            depth_path = os.path.join(self.tmp_dir, "depth", f"{i:04d}.npy")
            extr_path = os.path.join(self.tmp_dir, "extr", f"{i:04d}.npy")
            intr_path = os.path.join(self.tmp_dir, "intr", f"{i:04d}.npy")

            new_image, new_depth, new_K = process_view(image, depth, K)

            cv2.imwrite(image_path, cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR))
            np.save(depth_path, new_depth)
            np.save(intr_path, new_K)
            np.save(extr_path, extr)
        
        np.save(os.path.join(self.tmp_dir, "pcd.npy"), data["pcd"])
        
        with tarfile.open(self.tarfile, mode='a') as tar:
            tar.add(self.tmp_dir, arcname=f"{uid}")
