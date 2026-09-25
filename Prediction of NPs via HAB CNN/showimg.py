import cv2
import numpy as np


def show_cam_on_image(img, mask, use_rgb=True, colormap=cv2.COLORMAP_JET):
    """
    This function overlays the cam mask on the image as an heatmap.

    Parameters:
    img (np.array): The original image.
    mask (np.array): The cam mask.
    use_rgb (bool): Whether to use RGB or BGR color format for the final image.
    colormap (int): OpenCV colormap flag.

    Returns:
    cam (np.array): Original image with the heatmap overlayed.
    """
    # Normalize the mask to keep pixel values between 0 and 1
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)

    # Convert to RGB if needed
    if use_rgb:
        cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)

    return np.uint8(255 * cam)