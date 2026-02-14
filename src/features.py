# handcrafted features (baseline)
# Purpose: convert an image into numbers that correlate with quality. Sharpness (Laplacian variance): blurry images have fewer edges → lower variance.
# Contrast: low-contrast X-rays have a lower std.
# Brightness: too dark/bright can be flagged.
# Entropy: measures “information spread”; can drop with low contrast / extreme smoothing.
# SNR proxy (mean/std): crude, but useful baseline.

"""
Instead of giving the model 512 x 512 = 262,144 pixels, we summarize the image using a few meaningful numbers. We use 5 different features: 
1) sharpness which tells how clear the edges are
2) contrast, how spread out intensities are
3) brightness, how dark or bright overall
4) entropy, how much information/complexity (studied at school)
5) snr signal vs variation

Now the model works on structured data IMAGE ---> [sharpness, contrast, brightness, entropy, snr] 

In order to deal with the sharpness feature, I had to learn about the Laplacian Variance.
Here is what I learned:

The Laplacian operator is a mathematical filter that detects edges (Edges = zones where pixel intensity changes rapidly). For example black next to white ---> strong edge and
gray slowly changing ---> weak edge.
So it means that it detects areas where intensity changes abruptly. Blurry images have smoother transitions means weaker Laplacian response.

After applying the laplacian filter, we get a new image that highlights edges. From there: if the image is sharp: Many strong edges, 
high variation in Laplacian output, high variance whereas if the image is blurry: Few strong edges, values are small and smooth, low variance.

In short:
Sharp image = high Laplacian variance
Blurry image = low Laplacian variance




"""

import cv2
import numpy as np

def sharpness_laplacienne(img: np.ndarray) -> float:
    """Higher = sharper (less blurry)."""
    return float(cv2.Laplacian(img, cv2.CV_64F).var())

def contrast_std(img: np.ndarray) -> float:
    """Higher = more contrast (more intensity variation)."""
    return float(np.std(img))

def brightness_mean(img: np.ndarray) -> float:
    """Average brightness of the image."""
    return float(np.mean(img))

def entropy(img: np.ndarray) -> float:
    """
    Shannon entropy of grayscale histogram.
    Higher entropy often means more information variety.
    """
    hist = cv2.calcHist([img], [0], None, [256], [0, 256]).ravel()
    p = hist / (hist.sum() + 1e-12)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())

def snr_proxy(img: np.ndarray) -> float:
    """Very simple SNR-like metric: mean / std."""
    s = float(np.std(img)) + 1e-12
    return float(np.mean(img) / s)
