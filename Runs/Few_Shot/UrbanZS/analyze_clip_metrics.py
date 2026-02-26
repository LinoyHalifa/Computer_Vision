import csv
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image


from utils.common import NORMAL_LABELS, ANOMALY_LABELS
from utils.loaders import load_clip, precompute_text_embeddings


# ======================================================
# PATHS
# ======================================================
BASE_DIR = Path(
    r"C:\LogSAD\LogSAD-master\Anomaly Detection\UrbanZS"

)

EVAL_DIR = BASE_DIR / "evaluation"
NORMAL_DIR = EVAL_DIR / "normal"
ANOMALY_DIR = EVAL_DIR / "anomaly"

GT_CSV = EVAL_DIR / "gt_anomaly.csv"
OUT_CSV = EVAL_DIR / "analyze_clip_metrics.csv"


# ======================================================
# LABELS
# ======================================================
ALL_LABELS = NORMAL_LABELS + ANOMALY_LABELS
ALL_LABEL_NAMES = [name for (name, _) in ALL_LABELS]



# ======================================================
# LOAD GT
# ======================================================
def load_gt_labels(gt_csv_path):
    """
    Load GT labels into dict:
    { filename -> normal/anomaly_label }
    """
    gt_map = {}

    with open(gt_csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        print("GT CSV columns:", reader.fieldnames)
        for row in reader:
            filename = row["filename"].strip()
            gt_label_text = row["normal/anomaly_label"].strip()
            gt_map[filename] = gt_label_text

    return gt_map


# ======================================================
# INIT CLIP
# ======================================================
def init_clip():
    print("[INIT] CLIP...")
    clip_model, preprocess = load_clip()
    clip_model.eval()

    print("[INIT] Text embeddings...")
    all_text_embs = precompute_text_embeddings(clip_model, ALL_LABELS)


    return clip_model, preprocess, all_text_embs


# ======================================================
# ANALYZE ONE IMAGE (CLIP ONLY)
# ======================================================
def analyze_image(img_path, gt_label_text, clip_model, preprocess, all_text_embs):
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise ValueError(f"Cannot read image: {img_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    device = next(clip_model.parameters()).device
    img_pil = Image.fromarray(img_rgb)
    img_clip = preprocess(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        image_emb = clip_model.encode_image(img_clip)
        image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)

    image_emb_np = image_emb.detach().cpu().numpy()
    sims = (image_emb_np @ all_text_embs.T).squeeze(0)

    # sort similarities (descending)
    sorted_indices = np.argsort(sims)[::-1]

    predicted_label = ALL_LABEL_NAMES[sorted_indices[0]]

    gt_idx = ALL_LABEL_NAMES.index(gt_label_text)
    gt_similarity = float(sims[gt_idx])

    top1_correct = int(sorted_indices[0] == gt_idx)
    top3_correct = int(gt_idx in sorted_indices[:3])

    return predicted_label, gt_similarity, top1_correct, top3_correct


# ======================================================
# MAIN
# ======================================================
def main():
    print("\n==============================")
    print(" CLIP GLOBAL SEMANTIC EVALUATION")
    print("==============================\n")

    gt_map = load_gt_labels(GT_CSV)
    print(f"[INFO] Loaded GT labels for {len(gt_map)} images")

    clip_model, preprocess, all_text_embs = init_clip()

    rows = []

    for folder in [NORMAL_DIR, ANOMALY_DIR]:
        for img_path in sorted(folder.glob("*")):
            if not img_path.is_file():
                continue

            if img_path.name not in gt_map:
                print(f"[WARNING] No GT label for {img_path.name}, skipping")
                continue

            gt_label_text = gt_map[img_path.name]
            print(f"[ANALYZE] {img_path.name}")

            try:
                pred_label, gt_sim, top1_correct, top3_correct = analyze_image(
                    img_path,
                    gt_label_text,
                    clip_model,
                    preprocess,
                    all_text_embs,
                )
            except Exception as e:
                print(f"[ERROR] {img_path.name}: {e}")
                continue

            rows.append([
                img_path.name,
                gt_label_text,
                pred_label,
                gt_sim,
                top1_correct,
                top3_correct,
            ])

    # --- Save CSV ---
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "filename",
            "gt_label_text",
            "predicted_label",
            "gt_label_similarity",
            "top1_correct",
            "top3_correct",
        ])
        w.writerows(rows)

    # --- Metrics ---
    top1_corrects = [r[-2] for r in rows]
    top3_corrects = [r[-1] for r in rows]
    gt_sims = [r[-3] for r in rows]

    top1_accuracy = np.mean(top1_corrects)
    top3_accuracy = np.mean(top3_corrects)
    mean_gt_similarity = np.mean(gt_sims)

    print("\n=== CLIP GLOBAL METRICS ===")
    print(f"Top-1 Label Accuracy : {top1_accuracy:.4f}")
    print(f"Top-3 Label Accuracy : {top3_accuracy:.4f}")
    print(f"Mean GT Similarity   : {mean_gt_similarity:.4f}")

    print(f"\n[DONE] Results saved to: {OUT_CSV}")


if __name__ == "__main__":
    main()
