"""
Save rendering outputs to an uncompressed Tarball.
"""
import tarfile
import os
import shutil
import logging
import threading
from typing import Dict, List, Union

import cv2
import numpy as np
import psutil

from utils.postprocess import process_view


def log_resource_usage(resource_logger, interval):
    process = psutil.Process(os.getpid())
    cpu_percent = process.cpu_percent(interval=interval)
    cpu_times = process.cpu_times()
    mem_info = process.memory_info()

    resource_logger.info(
        f"CPU Usage - Percent: {cpu_percent:.2f}%, "
        f"User Time: {cpu_times.user:.2f}s, System Time: {cpu_times.system:.2f}s"
    )
    resource_logger.info(
        f"Memory Usage - RSS: {mem_info.rss / (1024 ** 2):.2f} MB, "
        f"VMS: {mem_info.vms / (1024 ** 2):.2f} MB\n"
    )


def periodic_resource_logger(log_file, interval):
    logger = logging.getLogger("resource_logger")
    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter("%(asctime)s - %(message)s")

    logger.setLevel(logging.INFO)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.propagate = False # Avoid polluting main logs

    while True:
        log_resource_usage(logger, interval)


class TarWriter():
    """
    Handles all rendering I/O operations, including saving outputs and logging.
    Saves data to a tarball to circumvent cluster file limits.
    Each shard is processed independently and has its own 'output/tmp/xxxxxx' directory, where the
    data is temporarily saved before being added to the shard tarball.
    """
    def __init__(self, output_dir: str, config_path: str, idx: int, log_resources: bool):
        self.idx = idx

        # Setup render logging
        log_dir = os.path.join(output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(log_dir, f"{idx:06d}.log")
        format = "%(asctime)s - %(levelname)s - %(message)s"
        logging.basicConfig(filename=log_file, level=logging.INFO, format=format)

        # Setup resource usage logging
        if log_resources:
            res_log_dir = os.path.join(output_dir, "resource_logs")
            os.makedirs(res_log_dir, exist_ok=True)

            res_log_file = os.path.join(res_log_dir, f"{idx:06d}.log")
            threading.Thread(
                target=periodic_resource_logger,
                args=(res_log_file, 60),
                daemon=True,
            ).start()

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
