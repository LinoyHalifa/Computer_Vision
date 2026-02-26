import csv
from pathlib import Path
import cv2

from utils.common import NORMAL_LABELS, ANOMALY_LABELS
from utils.loaders import load_clip, load_dino
import stations.station3_dino as station3

# ======================================================
# PATHS
# ======================================================
BASE_DIR = Path(r"C:\LogSAD\LogSAD-master\Anomaly Detection\UrbanZS")
EVAL_DIR = BASE_DIR / "evaluation"
NORMAL_DIR = EVAL_DIR / "normal"
ANOMALY_DIR = EVAL_DIR / "anomaly"

OUT_CSV = EVAL_DIR / "arch3_dino_clip_semantic_analysis.csv"

# ======================================================
# INIT MODELS
# ======================================================
def init_models():
    print("[INIT] DINO...")
    dino_model, dino_transform = load_dino()

    # ⬅️ תיקון 1: הקריאה דרך station3
    normal_patch_bank = station3.precompute_dino_normal_patch_bank(
        dino_model, dino_transform
    )

    print("[INIT] CLIP...")
    clip_model, clip_preprocess = load_clip()

    return (
        dino_model,
        dino_transform,
        normal_patch_bank,
        clip_model,
        clip_preprocess,
    )

# ======================================================
# ANALYZE ONE IMAGE (DINO + CLIP STATION)
# ======================================================
def analyze_image(
    img_path,
    dino_model,
    dino_transform,
    normal_patch_bank,
    clip_model,
    clip_preprocess,
):
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise ValueError(f"Cannot read image: {img_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # ⬅️ תיקון 2: קריאה דרך station3
    score_dino, score_clip_normal, score_clip_anomaly = station3.station_dino_with_clip(
        img_rgb,
        dino_model,
        dino_transform,
        normal_patch_bank,
        clip_model,
        clip_preprocess,
        NORMAL_LABELS,
        ANOMALY_LABELS,
    )

    # semantic margin (same idea as CLIP-only)
    semantic_margin = score_clip_anomaly - score_clip_normal

    return semantic_margin

# ======================================================
# MAIN
# ======================================================
def main():
    (
        dino_model,
        dino_transform,
        normal_patch_bank,
        clip_model,
        clip_preprocess,
    ) = init_models()

    rows = []

    for label_name, folder, gt in [
        ("normal", NORMAL_DIR, 0),
        ("anomaly", ANOMALY_DIR, 1),
    ]:
        for img_path in sorted(folder.glob("*")):
            if not img_path.is_file():
                continue

            print(f"[ANALYZE] {img_path.name}")

            try:
                margin = analyze_image(
                    img_path,
                    dino_model,
                    dino_transform,
                    normal_patch_bank,
                    clip_model,
                    clip_preprocess,
                )
            except Exception as e:
                print(f"[ERROR] {img_path.name}: {e}")
                continue

            rows.append([
                img_path.name,
                label_name,
                gt,
                margin,
            ])

    # ==================================================
    # SAVE CSV
    # ==================================================
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "filename",
            "gt_label",
            "gt_binary",
            "semantic_margin",
        ])
        w.writerows(rows)

    print(f"\n[DONE] Saved ARCH3 DINO+CLIP semantic analysis to: {OUT_CSV}")

if __name__ == "__main__":
    main()
