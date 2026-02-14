
# Here are perosnal notes, the material I went through (I learned on my own) in order to write that code.

"""
A normal image (color image) has 3 channels: Red, Green, Blue. So each pixel is something like: [R, G, B]. So a color image is actually a 3D array: height × width × 3.
Grayscale means that instead of 3 color values per pixel, we keep only 1 intensity value per pixel. Intensity is between 0 and 255. With 0 = black, 255 = white and 128 = Gray. That way
we now manipulate a 2D array with height x width.
Why do we use grayscale? Because: X-ray images are already mostly grayscale, we don’t need color to measure blur, contrast, entropy. That simplifies the data, it reduces computation.
and we remove useless information.

I also have to resize each images so we make sure that every image has the same height and width so the comparison is reliable If image sizes are different: features are inconsistent
CNN cannot work, comparison becomes unreliable

"""

# The purpose is to make all the images consistent

import cv2
import numpy as np

IMG_SIZE = (512, 512)  # (width, height)

def load_grayscale(path: str) -> np.ndarray:
    # Read an image from disk as grayscale (2D array).
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Problem with the image: {path}")
    return img

def preprocess(img_gray: np.ndarray) -> np.ndarray:

    # Standardizing the image:
    # resizing to IMG_SIZE
    # keep uint8 (0..255) for the feature extraction
    return cv2.resize(img_gray, IMG_SIZE, interpolation=cv2.INTER_AREA)

