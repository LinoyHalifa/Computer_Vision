import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

scored_path = Path(r"C:\LogSAD\LogSAD-master\Urban_Risk\UrbanRiskGPT\results\gpt_only\reasoning_scored.jsonl")

scores = []

with scored_path.open("r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        if "reasoning_score" in row:
            scores.append(row["reasoning_score"])

scores = np.array(scores)
print("Total scored samples:", len(scores))  # ← להוסיף את זה


plt.figure(figsize=(7,5))

plt.hist(scores, bins=np.arange(1,12)-0.5)

plt.title("Score distribution for reasoning\n1-10", fontsize=16)
plt.xlabel("Reasoning score", fontsize=14)
plt.ylabel("Score distribution (incidence)", fontsize=14)

plt.xticks(range(1,11))
plt.grid(alpha=0.3)

plt.savefig(r"C:\LogSAD\LogSAD-master\Urban_Risk\UrbanRiskGPT\results\gpt_only\reasoning_distribution.png", dpi=200)
plt.show()
