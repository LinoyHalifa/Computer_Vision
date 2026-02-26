import json
from pathlib import Path
import cv2
import numpy as np

from utils.loaders import load_dino
from stations.station3_dino import (
    station_dino_patches,
    precompute_dino_normal_patch_bank
)

# --------------------------------------------------------
# CONFIG
# --------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
NORMAL_DIR = BASE_DIR / "normal_images"   # תמונות נורמליות בלבד
OUT_FILE = BASE_DIR / "statistics.json"


def compute_stats():
    print("[INFO] Loading DINO...")

    dino_model, dino_transform = load_dino()
    dino_bank = precompute_dino_normal_patch_bank(
        dino_model, dino_transform
    )

    print("[INFO] Running DINO statistics on normal images...")

    dino_scores = []

    for fname in NORMAL_DIR.iterdir():
        if fname.suffix.lower() not in [".jpg", ".png", ".jpeg"]:
            continue

        img_bgr = cv2.imread(str(fname))
        if img_bgr is None:
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        score_dino = station_dino_patches(
            img_rgb,
            dino_model,
            dino_transform,
            dino_bank,
        )

        dino_scores.append(score_dino)

    if len(dino_scores) == 0:
        raise RuntimeError("No valid normal images found.")

    stats = {
        "dino": {
            "mean": float(np.mean(dino_scores)),
            "std": float(np.std(dino_scores))
        }
    }

    with open(OUT_FILE, "w") as f:
        json.dump(stats, f, indent=4)

    print("\n[INFO] Saved statistics to:", OUT_FILE)
    print(json.dumps(stats, indent=4))


if __name__ == "__main__":
    compute_stats()
