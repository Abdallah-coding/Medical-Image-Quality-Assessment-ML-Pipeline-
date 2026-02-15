# Purpose: run features on every image and build a table

from pathlib import Path
import pandas as pd


from .dataset import load_paths_and_labels
from .preprocessing import load_grayscale, preprocess
from .features import (sharpness_laplacienne, contrast_std, brightness_mean, entropy, snr_proxy)


def main():

    Path("reports").mkdir(exist_ok=True)
    
    paths, y = load_paths_and_labels()
    if len(paths) == 0:
        raise SystemExit("No images found in data/generated/. Run build_dataset first.")

    rows = []
    for p, label in zip(paths, y):
        im = preprocess(load_grayscale(p))

        rows.append({
            "path": p,
            "label": int(label),
            "sharpness": sharpness_laplacienne(im),
            "contrast": contrast_std(im),
            "brightness": brightness_mean(im),
            "entropy": entropy(im),
            "snr": snr_proxy(im),
        })

    df = pd.DataFrame(rows)
    df.to_csv("reports/features.csv", index=False)
    print("features.csv done,", len(df), "images processed")


if __name__ == "__main__":
    main()
