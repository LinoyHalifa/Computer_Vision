import json
from pathlib import Path
from openai import OpenAI
import time

# ===============================
# PATHS
# ===============================
results_path = Path(r"C:\LogSAD\LogSAD-master\Urban_Risk\UrbanRiskGPT\results\gpt_only/results_night.jsonl")
output_path  = Path(r"C:\LogSAD\LogSAD-master\Urban_Risk\UrbanRiskGPT\results\gpt_only\reasoning_scored.jsonl")

client = OpenAI()

# ===============================
# PROMPT FOR SCORING
# ===============================
def score_reasoning(text):

    prompt = f"""
You are evaluating the quality of an AI explanation.

Score the reasoning from 1 to 10 based on:
- semantic correctness
- contextual understanding
- clarity
- safety relevance

Return ONLY a number between 1 and 10.

Explanation:
{text}
"""

    resp = client.chat.completions.create(
        model="gpt-5.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return int(resp.choices[0].message.content.strip())

# ===============================
# RUN
# ===============================
with results_path.open("r", encoding="utf-8") as f, \
     output_path.open("w", encoding="utf-8") as out:

    for line in f:
        row = json.loads(line)

        if "result" not in row:
            continue

        anomalies = row["result"].get("anomalies", [])

        if len(anomalies) == 0:
            continue

        # take first anomaly reasoning
        reasoning = anomalies[0].get("reasoning", "")

        if reasoning == "":
            continue

        score = score_reasoning(reasoning)

        row["reasoning_score"] = score

        out.write(json.dumps(row) + "\n")

        print("Score:", score)

        time.sleep(0.2)  # avoid rate limit
