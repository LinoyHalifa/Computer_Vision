import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# ===== path to your results file =====
results_path = Path(r"C:\LogSAD\LogSAD-master\Urban_Risk\UrbanRiskGPT\results\gpt_only\results_night.jsonl")

scores = []

with results_path.open("r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)

        if "result" not in row:
            continue

        score = row["result"].get("overall_risk_score_1_to_10")
        if score is not None:
            scores.append(int(score))

scores = np.array(scores)

print("Num samples:", len(scores))
print("Mean:", scores.mean())
print("Median:", np.median(scores))
print("Std:", scores.std())

# ===== EXACT STYLE LIKE THE PAPER =====
# Count how many images got each score
unique, counts = np.unique(scores, return_counts=True)

plt.figure(figsize=(7,5))
plt.bar(unique, counts, width=0.6)

plt.xlabel("Anomaly score")
plt.ylabel("Number of anomalies detected")
plt.title("Anomalies distribution")

plt.xticks(range(1,11))
plt.grid(axis="y", alpha=0.3)

plt.savefig(r"C:\LogSAD\LogSAD-master\Urban_Risk\UrbanRiskGPT\results\gpt_only\anomaly_distribution_bar.png", dpi=200)
plt.show()
