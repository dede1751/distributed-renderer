import os

import numpy as np
from PIL import Image
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip


def generate_video(out_file, images, extrinsics, frames_per_image=1, fps=30):
    print(f"Generating clip from {len(images)} views.")

    # Sort images by azimuth to rotate around the object
    azimuths = [np.arctan2(matrix[0][3], matrix[1][3]) for matrix in extrinsics]
    sorted_indices = np.argsort(azimuths)
    sorted_images = [images[i] for i in sorted_indices]

    # Duplicate images over frames_per_image frames
    expanded_images = [img for img in sorted_images for _ in range(frames_per_image)]
    print(f"Generating video with {len(expanded_images)} frames.")

    clip = ImageSequenceClip(expanded_images, fps=fps)
    clip.write_videofile(out_file, codec="libx264")
    print(f"Video saved as {out_file}")


def generate_all_videos(in_dir, out_dir, frames_per_image=1, max_views=None):
    """
    Generate videos for all objects within 'in_dir'.
    This directory is assumed to be structured like the rendering outputs:

    in_dir
    ├── object_1
    │   ├── images
    │   │   ├── 0000.png
    │   │   ├── 0001.png
    │   │   └── ...
    │   └── extr
    │       ├── 0000.npy
    │       ├── 0001.npy
    │       └── ...
    ├── object_2
    │   └── ...
    └── ...
    """
    os.makedirs(out_dir, exist_ok=True)
    uids = os.listdir(in_dir)
    for uid in uids:
        img_path = os.path.join(in_dir, uid, "images")
        extr_path = os.path.join(in_dir, uid, "extr")
        images = [np.array(Image.open(os.path.join(img_path, fname))) for fname in os.listdir(img_path)]
        extrinsics = [np.load(os.path.join(extr_path, fname)) for fname in os.listdir(extr_path)]

        if max_views is not None:
            images = images[:max_views]
            extrinsics = extrinsics[:max_views]

        print(f"\n[+] Generating video for {uid} with {len(images)} views.")
        out_file = os.path.join(out_dir, f"{uid}.mp4")
        generate_video(out_file, images, extrinsics, frames_per_image)
