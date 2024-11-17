"""
Save rendering outputs to an uncompressed Tarball.
"""
from blenderproc.python.utility.MathUtility import MathUtility

import tarfile
import os
import shutil
import logging
from typing import Dict, List, Union

import numpy as np
import cv2
import trimesh
    

class TarWriter():
    def __init__(self, output_dir: str, config_path: str, idx: int):
        self.idx = idx
        self.tmp_dir = os.path.join(output_dir, "tmp", f"{idx:06d}")
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        os.makedirs(self.tmp_dir, exist_ok=True)

        for cat in ["images", "depth", "extr"]:
            os.makedirs(os.path.join(self.tmp_dir, cat))

        tar_dir = os.path.join(output_dir, "shards")
        os.makedirs(tar_dir, exist_ok=True)

        self.tarfile = os.path.join(tar_dir, f"shard_{idx:06d}.tar")
        with tarfile.open(self.tarfile, mode='w') as tar:
            tar.add(config_path, arcname="config.json")
        logging.info(f"Writing data to {self.tarfile}")
        
        #self.blender2opencv = MathUtility.build_coordinate_frame_changing_transformation_matrix(["X", "-Y", "-Z"])
    
    def write(self, objaverse_id: str, data: Dict[str, Union[np.ndarray, List[np.ndarray], trimesh.PointCloud]]):
        for i in range(data["num_views"]):
            image = data["colors"][i]
            image_path = os.path.join(self.tmp_dir, "images", f"{i:04d}.png")
            cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA))

            depth = data["depth"][i]
            depth_path = os.path.join(self.tmp_dir, "depth", f"{i:04d}.npy")
            np.save(depth_path, depth)

            extr = data["extr"][i]
            np.save(os.path.join(self.tmp_dir, "extr", f"{i:04d}.npy"), extr)
        
        #data["pcd"].apply_transform(self.blender2opencv)
        data["pcd"].export(os.path.join(self.tmp_dir, "pcd.ply"))
        np.save(os.path.join(self.tmp_dir, "intr.npy"), data["intr"])
        
        with tarfile.open(self.tarfile, mode='a') as tar:
            tar.add(self.tmp_dir, arcname=f"{objaverse_id}")
