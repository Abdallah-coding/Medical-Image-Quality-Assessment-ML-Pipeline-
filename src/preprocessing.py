# common preprocessing


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

