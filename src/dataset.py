# Purpose: list all generated images and return: file paths, labels (good=1, poor=0)

from pathlib import Path
from typing import List, Tuple
import numpy as np

GEN_DIR = Path("data/generated")
SUPPORTED = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _grab(folder: Path) -> List[Path]:
    return [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED]


def load_paths_and_labels() -> Tuple[List[str], np.ndarray]:
    good = _grab(GEN_DIR / "good")
    poor = _grab(GEN_DIR / "poor")

    paths = [str(p) for p in good] + [str(p) for p in poor]
    y = np.array([1] * len(good) + [0] * len(poor), dtype=np.int64)

    return paths, y
