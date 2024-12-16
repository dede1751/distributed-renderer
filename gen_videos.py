import argparse
import os

import numpy as np
from PIL import Image
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip


FPS = 30

def get_azimuth(matrix):
    pos = matrix[:3, 3]
    azimuth = np.arctan2(pos[0], pos[1])
    return azimuth

def generate_video(out_file, images, extrinsics, frames_per_image):
    print(f"Generating clip from {len(images)} views.")

    # Sort images by azimuth to rotate around the object
    azimuths = [np.arctan2(matrix[0][3], matrix[1][3]) for matrix in extrinsics]
    sorted_indices = np.argsort(azimuths)
    sorted_images = [images[i] for i in sorted_indices]

    # Duplicate images over frames_per_image frames
    expanded_images = [img for img in sorted_images for _ in range(frames_per_image)]
    print(f"Generating video with {len(expanded_images)} frames.")

    clip = ImageSequenceClip(expanded_images, fps=FPS)
    clip.write_videofile(out_file, codec="libx264")
    print(f"Video saved as {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make a video of the rendered images for a single object.")
    parser.add_argument('--path', type=str, required=True, help="Path to the object's rendering outputs (final directory should be a UID).")
    parser.add_argument('--out', type=str, required=True, help="Name of the output '.mp4' clip.")
    parser.add_argument('--fpi', type=int, default=1, help="Number of frames to display individual views for.")
    parser.add_argument('--max_views', type=int, default=None, help="Maximum number of views to load.")
    args = parser.parse_args()

    img_path = os.path.join(args.path, "images")
    extr_path = os.path.join(args.path, "extr")
    images = [np.array(Image.open(os.path.join(img_path, fname))) for fname in os.listdir(img_path)]
    extrinsics = [np.load(os.path.join(extr_path, fname)) for fname in os.listdir(extr_path)]

    if args.max_views is not None:
        images = images[:args.max_views]
        extrinsics = extrinsics[:args.max_views]

    generate_video(args.out, images, extrinsics, args.fpi)
