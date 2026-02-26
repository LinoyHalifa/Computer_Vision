import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    auc
)

# ===============================
# PATH
# ===============================
results_path = Path(r"C:\LogSAD\LogSAD-master\Urban_Risk\UrbanRiskGPT\results\gpt_only/results_night.jsonl")
manifest_path = Path(r"C:\LogSAD\LogSAD-master\Urban_Risk\UrbanRiskGPT\data\manifest\manifest_night.jsonl")



# ===============================
# LOAD DATA
# ===============================

# ===============================
# LOAD GT FROM MANIFEST
# ===============================
label_map = {}

with manifest_path.open("r", encoding="utf-8") as f:
    for line in f:
        m = json.loads(line)
        label_map[m["sample_id"]] = m["label"]


scores = []
gt_labels = []

with results_path.open("r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)

        if "result" not in row:
            continue

        sample_id = row["sample_id"]
        gt = label_map[sample_id]

        score = row["result"].get("overall_risk_score_1_to_10")

        if score is not None:
            scores.append(score)
            gt_labels.append(gt)

scores = np.array(scores)
gt_labels = np.array(gt_labels)

print("Num samples:", len(scores))

# ===============================
# THRESHOLD
# ===============================
threshold = 6  # אפשר לשנות אחר כך

pred_labels = (scores >= threshold).astype(int)

# ===============================
# METRICS
# ===============================
acc = accuracy_score(gt_labels, pred_labels)
prec = precision_score(gt_labels, pred_labels)
rec = recall_score(gt_labels, pred_labels)
f1 = f1_score(gt_labels, pred_labels)

roc_auc = roc_auc_score(gt_labels, scores)

precision_curve, recall_curve, _ = precision_recall_curve(gt_labels, scores)
pr_auc = auc(recall_curve, precision_curve)

print("\n=== ZERO-SHOT METRICS ===")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1:        {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")
print(f"PR-AUC:    {pr_auc:.4f}")
print(f"Threshold: {threshold}")

# ===============================
# CONFUSION MATRIX
# ===============================
cm = confusion_matrix(gt_labels, pred_labels)

plt.figure(figsize=(5,5))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix (Zero-Shot)")
plt.colorbar()

plt.xticks([0,1], ["Pred Normal","Pred Anomaly"])
plt.yticks([0,1], ["True Normal","True Anomaly"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i,j], ha="center", va="center", color="black")

plt.tight_layout()
plt.savefig(r"C:\LogSAD\LogSAD-master\Urban_Risk\UrbanRiskGPT\results\gpt_only//confusion_matrix_zero_shot.png", dpi=200)
plt.show()


metrics_table = pd.DataFrame([{
    "Approach": "Zero-Shot",
    "roc_auc": roc_auc,
    "pr_auc": pr_auc,
    "accuracy": acc,
    "precision": prec,
    "recall": rec,
    "f1": f1,
    "Thr": threshold
}])

print("\n=== METRICS TABLE ===")
print(metrics_table)

metrics_table.to_csv(
    r"C:\LogSAD\LogSAD-master\Urban_Risk\UrbanRiskGPT\results\gpt_only\zero_shot_metrics.csv",
    index=False
)


# ===============================
# ROC CURVE
# ===============================
fpr, tpr, thresholds = roc_curve(gt_labels, scores)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
plt.plot([0,1], [0,1], linestyle="--")  # random model line

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Zero Shot")
plt.legend()
plt.grid(alpha=0.3)

plt.savefig(r"C:\LogSAD\LogSAD-master\Urban_Risk\UrbanRiskGPT\results\gpt_only\roc_curve_zero_shot.png", dpi=200)
plt.show()

