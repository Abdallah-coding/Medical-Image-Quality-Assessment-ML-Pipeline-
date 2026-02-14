# dataset loader (images + labels)

# X-ray datasets usually don’t label “quality”, so we simulate poor acquisition: good = original preprocessed image, poor = blur / noise / low-contrast versions


import random
from pathlib import Path

import cv2
import numpy as np

from .preprocessing import load_grayscale, preprocess

RAW_DIR = Path("data/raw")
GEN_DIR = Path("data/generated")
SUPPORTED = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def pick_images(folder: Path):
    return [x for x in folder.iterdir() if x.is_file() and x.suffix.lower() in SUPPORTED]


def blur(im: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(im, (11, 11), 0)


def noise(im: np.ndarray, s: float = 18.0) -> np.ndarray:
    n = np.random.normal(0, s, im.shape).astype(np.float32)
    out = im.astype(np.float32) + n
    return np.clip(out, 0, 255).astype(np.uint8)


def low_contrast(im: np.ndarray) -> np.ndarray:
    f = im.astype(np.float32)
    out = 128 + 0.5 * (f - 128)
    return np.clip(out, 0, 255).astype(np.uint8)


def dump(im: np.ndarray, where: Path):
    where.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(where), im)


def main(n_max: int = 200, seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    files = pick_images(RAW_DIR)
    if not files:
        raise SystemExit("No images in data/raw/")

    random.shuffle(files)
    files = files[:n_max]

    good = GEN_DIR / "good"
    bad = GEN_DIR / "poor"
    good.mkdir(parents=True, exist_ok=True)
    bad.mkdir(parents=True, exist_ok=True)

    for f in files:
        im = preprocess(load_grayscale(str(f)))

        dump(im, good / f"{f.stem}_good.png")
        dump(blur(im), bad / f"{f.stem}_blur.png")
        dump(noise(im), bad / f"{f.stem}_noise.png")
        dump(low_contrast(im), bad / f"{f.stem}_lowcontrast.png")

    print("done:", len(files), "raw ->",
          len(list(good.glob("*.png"))), "good /",
          len(list(bad.glob("*.png"))), "poor")


if __name__ == "__main__":
    main()
