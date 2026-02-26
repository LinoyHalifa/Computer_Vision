import os
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    precision_recall_fscore_support,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns

# ======================================================
# PROJECT IMPORTS – ONLY STATION 3
# ======================================================

from utils.common import NORMAL_LABELS, ANOMALY_LABELS
from utils.loaders import load_clip, load_dino
from stations.station3_dino import (
    precompute_dino_normal_patch_bank,
    station_dino_with_clip,
    compute_final_score
)

# ======================================================
# PATHS
# ======================================================

BASE_DIR = Path(r"C:\LogSAD\LogSAD-master\Anomaly Detection\UrbanZS")

EVAL_DIR = BASE_DIR / "evaluation"
NORMAL_DIR = EVAL_DIR / "normal"
ANOMALY_DIR = EVAL_DIR / "anomaly"

GT_CSV_PATH = EVAL_DIR / "gt.csv"
RESULTS_CSV_PATH = EVAL_DIR / "results.csv"
METRICS_JSON_PATH = EVAL_DIR / "metrics.json"

# ======================================================
# GT FILE
# ======================================================

def build_gt(normal_dir, anomaly_dir, out_csv):
    rows = []

    for p in sorted(normal_dir.glob("*")):
        if p.is_file():
            rows.append((p.name, 0))

    for p in sorted(anomaly_dir.glob("*")):
        if p.is_file():
            rows.append((p.name, 1))

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "label"])
        for fn, lbl in rows:
            w.writerow([fn, lbl])

    print(f"[GT] wrote {len(rows)} rows.")


def load_gt(path):
    gt = {}
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            gt[row["filename"]] = int(row["label"])
    return gt


# ======================================================
# INIT MODELS (ONLY CLIP + DINO)
# ======================================================

def init_models():
    print("[INIT] CLIP...")
    clip_model, clip_preprocess = load_clip()

    print("[INIT] DINO...")
    dino_model, dino_transform = load_dino()

    print("[INIT] DINO patch bank...")
    normal_patch_bank = precompute_dino_normal_patch_bank(
        dino_model, dino_transform
    )

    return {
        "clip_model": clip_model,
        "clip_preprocess": clip_preprocess,
        "dino": dino_model,
        "dino_transform": dino_transform,
        "normal_patch_bank": normal_patch_bank,
    }



# ======================================================
# RUN PIPELINE ON ONE IMAGE (STATION 3 ONLY)
# ======================================================

def run_pipeline(path, M):
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        raise ValueError(f"cannot read {path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # --- Station 3: DINO + CLIP ---
    score_dino, score_clip_normal, score_clip_anomaly = station_dino_with_clip(
        img_rgb=img_rgb,
        dino_model=M["dino"],
        dino_transform=M["dino_transform"],
        normal_patch_bank=M["normal_patch_bank"],
        clip_model=M["clip_model"],
        clip_preprocess=M["clip_preprocess"],
        normal_labels=NORMAL_LABELS,
        anomaly_labels=ANOMALY_LABELS,
    )

    final_score = compute_final_score(
        score_dino,
        score_clip_normal,
        score_clip_anomaly
    )

    return final_score, score_dino, score_clip_normal, score_clip_anomaly


# ======================================================
# METRICS
# ======================================================

def compute_metrics(y_true, y_score):
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    # ROC
    fpr, tpr, roc_thr = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    best_idx = np.argmax(tpr - fpr)
    best_thr = roc_thr[best_idx]

    y_pred = (y_score >= best_thr).astype(int)

    # PR
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall_curve, precision_curve)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "thr": float(best_thr),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "precision_curve": precision_curve.tolist(),
        "recall_curve": recall_curve.tolist(),
    }


# ======================================================
# MAIN
# ======================================================

def main():

    if not GT_CSV_PATH.exists():
        build_gt(NORMAL_DIR, ANOMALY_DIR, GT_CSV_PATH)

    gt = load_gt(GT_CSV_PATH)

    M = init_models()

    y_true = []
    y_score = []

    with open(RESULTS_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "filename",
            "gt",
            "final_score",
            "score_dino",
            "score_clip_normal",
            "score_clip_anomaly"
        ])

        for fn, lbl in gt.items():
            img_path = NORMAL_DIR / fn if lbl == 0 else ANOMALY_DIR / fn
            print(f"[EVAL] {fn}")

            try:
                final_s, dino_s, clip_norm_s, clip_anom_s = run_pipeline(img_path, M)
            except Exception as e:
                print(f"[ERROR] {fn}: {e}")
                continue

            y_true.append(lbl)
            y_score.append(final_s)

            w.writerow([
                fn,
                lbl,
                final_s,
                dino_s,
                clip_norm_s,
                clip_anom_s
            ])

    if len(y_true) == 0:
        print("[EVAL] No images processed.")
        return

    metrics = compute_metrics(y_true, y_score)

    with open(METRICS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("\n===== METRICS =====")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # ------------------ ROC ------------------
    plt.figure()
    plt.plot(metrics["fpr"], metrics["tpr"], label=f"AUC = {metrics['roc_auc']:.3f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.grid(True)
    plt.legend()
    plt.savefig(EVAL_DIR / "roc_curve.png", dpi=300)
    plt.close()

    # ------------------ PR ------------------
    plt.figure()
    plt.plot(metrics["recall_curve"], metrics["precision_curve"],
             label=f"PR-AUC = {metrics['pr_auc']:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True)
    plt.legend()
    plt.savefig(EVAL_DIR / "pr_curve.png", dpi=300)
    plt.close()

    # ------------------ Confusion Matrix ------------------
    cm = np.array([[metrics["tn"], metrics["fp"]],
                   [metrics["fn"], metrics["tp"]]])

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred Normal", "Pred Anomaly"],
                yticklabels=["True Normal", "True Anomaly"])
    plt.title("Confusion Matrix")
    plt.savefig(EVAL_DIR / "confusion_matrix.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
