import numpy as np
import cv2

def square_center_crop(
    image: np.ndarray, depth: np.ndarray, mask: np.ndarray,
    crop_size: int = 224, border: int = 10, resize: bool = True
):
    """
    Crop an image and the corresponding depth map using the given mask.
    If the crop would intersect the mask, resize the image to fit.
    """
    x, y, w, h = cv2.boundingRect(mask)
    image_h, image_w = image.shape[:2]

    # Add border to bbox
    x_expanded = max(0, x - border)
    y_expanded = max(0, y - border)
    w_expanded = min(w + 2 * border, image_w - x_expanded)
    h_expanded = min(h + 2 * border, image_h - y_expanded)

    # Shrink image so that bbox+border fit within the crop
    scale_factor = 1
    if resize and (w_expanded > crop_size or h_expanded > crop_size):
        scale_factor = crop_size / max(w_expanded, h_expanded)

        # Resize the original image and mask
        new_width = int(image_w * scale_factor)
        new_height = int(image_h * scale_factor)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        depth = cv2.resize(depth, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

        # Update the bounding box coordinates after resizing
        x = int(x * scale_factor)
        y = int(y * scale_factor)
        w = int(w * scale_factor)
        h = int(h * scale_factor)

    # Use bbox center as crop center
    center_x = x + w // 2
    center_y = y + h // 2
    crop_x = np.clip(center_x - crop_size // 2, 0, image_w - crop_size)
    crop_y = np.clip(center_y - crop_size // 2, 0, image_h - crop_size)

    image = image[crop_y:crop_y + crop_size, crop_x:crop_x + crop_size]
    depth = depth[crop_y:crop_y + crop_size, crop_x:crop_x + crop_size]

    return image, depth, scale_factor, (crop_x, crop_y)


def colmap_to_opencv_intrinsics(K):
    """
    Modify camera intrinsics to follow a different convention.
    Coordinates of the center of the top-left pixels are by default:
    - (0.5, 0.5) in Colmap
    - (0,0) in OpenCV
    """
    K[0, 2] -= 0.5
    K[1, 2] -= 0.5
    return K

def scale_intrinsics(K: np.ndarray, scale_factor: float, crop: (int, int)):
    """
    Scale camera intrinsics to account for the crop.
    """
    K[:2, :3] *= scale_factor
    K[0, 2] -= crop[0]
    K[1, 2] -= crop[1]
    return K

def process_view(image, depth, K):
    image, depth, K = image.copy(), depth.copy(), K.copy()

    depth[depth > 65535] = 0
    mask = (depth > 0).astype(np.uint8)
    image, depth, scale_factor, crop = square_center_crop(image, depth, mask)
    
    # Mask the background again on the cropped image
    image[depth == 0] = [255, 255, 255]

    # Update intrinsics, first converting to opencv convention
    K = colmap_to_opencv_intrinsics(K)
    K = scale_intrinsics(K, scale_factor, crop)
    return image, depth, K
