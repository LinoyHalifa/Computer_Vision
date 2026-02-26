import pandas as pd
import numpy as np

# ======================================================
# LOAD DATA
# ======================================================
df = pd.read_csv(
    r"C:\LogSAD\LogSAD-master\Anomaly Detection\UrbanZS\evaluation\arch3_dino_clip_semantic_analysis.csv"
)

df = df.dropna(subset=["semantic_margin"])

# ======================================================
# COMPUTE NORMALIZATION FROM THIS DATASET
# ======================================================
mean = df["semantic_margin"].mean()
std = df["semantic_margin"].std()

# ======================================================
# NORMALIZE + SIGMOID
# ======================================================
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

df["semantic_z"] = (df["semantic_margin"] - mean) / std
df["semantic_score"] = sigmoid(df["semantic_z"])

# ======================================================
# SPLIT GROUPS
# ======================================================
normal = df[df.gt_binary == 0]["semantic_score"]
anomaly = df[df.gt_binary == 1]["semantic_score"]

# ======================================================
# BASIC STATS
# ======================================================
stats = {}

stats["mean_normal"] = normal.mean()
stats["mean_anomaly"] = anomaly.mean()

stats["std_normal"] = normal.std()
stats["std_anomaly"] = anomaly.std()

stats["median_normal"] = normal.median()
stats["median_anomaly"] = anomaly.median()

# ======================================================
# SEPARATION GAP
# ======================================================
stats["mean_gap"] = stats["mean_anomaly"] - stats["mean_normal"]

# ======================================================
# SIGN AGREEMENT
# ======================================================
stats["sign_agreement_anomaly"] = (anomaly > 0.5).mean()
stats["sign_agreement_normal"] = (normal < 0.5).mean()

# ======================================================
# EFFECT SIZE (Cohen's d)
# ======================================================
pooled_std = np.sqrt(
    ((len(anomaly) - 1) * anomaly.std()**2 +
     (len(normal) - 1) * normal.std()**2)
    / (len(anomaly) + len(normal) - 2)
)

stats["cohens_d"] = stats["mean_gap"] / pooled_std

# ======================================================
# PRINT RESULTS
# ======================================================
print("\n===== NORMALIZED DINO+CLIP SEMANTIC ANALYSIS – ARCHITECTURE 3 =====")
print(f"(normalization mean={mean:.6f}, std={std:.6f})")
for k, v in stats.items():
    print(f"{k}: {v:.4f}")
