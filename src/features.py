# handcrafted features (baseline)
# !! First I explain all the materials I learned and then below I put the code !!

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


Contrast measures how spread out pixel values are.

If most pixels are around 120–130:
--> low contrast
--> small standard deviation

If pixels range from 10 to 240:
--> high contrast
--> large standard deviation


Brightness = average pixel value.
If mean is:
very low --> image is dark
very high --> image is too bright



Here entropy measures how unpredictable pixel intensities are.

If an image has:

many different intensity levels evenly distributed ---> high entropy
if most pixels are similar ---> low entropy

Low entropy often means: strong smoothing, very low contrast, information loss


SNR = Signal-to-Noise Ratio. We approximative SNR ≈ mean / standard deviation, so noisy images tend to have lower SNR proxy. (The average of all pixel intensities in the image)
If most pixels are close to the mean:
- low std

If pixels vary a lot:
- high std


Finally an important point to understand how we process and why these features:
Noise:
increases variation metrics
decreases SNR

Blur:
decreases Laplacian strongly
slightly decreases entropy
slightly decreases contrast

"""

import cv2
import numpy as np

def sharpness_laplacienne(img: np.ndarray) -> float:
    return float(cv2.Laplacian(img, cv2.CV_64F).var())

def contrast_std(img: np.ndarray) -> float:
    return float(np.std(img))

def brightness_mean(img: np.ndarray) -> float:
    return float(np.mean(img))

def entropy(img: np.ndarray) -> float:
    """
    We compute a histogram because:
    Entropy requires a probability distribution.
    The histogram gives:
    How often each intensity value appears.
    Then we convert counts to probabilities
    """
    
    hist = cv2.calcHist([img], [0], None, [256], [0, 256]).ravel()
    p = hist / (hist.sum() + 1e-12)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())

def snr_proxy(img: np.ndarray) -> float:
    # Here we don't use a true SNR in fact true SNR requires separating signal and noise which is complex
    s = float(np.std(img)) + 1e-12
    return float(np.mean(img) / s)
