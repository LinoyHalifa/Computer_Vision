import json
import re
import numpy as np

from utils.common import GPT_MODEL
from utils.loaders import build_openai_client
from utils.common import USE_STATION4



# ----------------------------------------------------------
# Helper: fallback extraction of a number between 0–1
# ----------------------------------------------------------
def extract_number_fallback(text):
    matches = re.findall(r"0\.\d+|1\.0+", text)
    if matches:
        num = float(matches[0])
        return float(np.clip(num, 0.0, 1.0))
    return 0.0


# ----------------------------------------------------------
# Station 4: GPT Logic Reasoning
# ----------------------------------------------------------
def gpt_logic_score(
        anomaly_label,
        anomaly_text,
        model_scores,        # dict: {"yolo_clip": , "sam_clip": , "dino": }
        objects_summary
):
    """
    Station 4:
    Logical reasoning with GPT — mathematically calibrated.

    Steps:
    ------
    1. GPT produces a raw logic score L_raw in [0,1].
    2. Visual confidence C_v = max(yolo_clip, sam_clip, dino).
    3. Final logic score:
            L_final = C_v * L_raw

       This mirrors Equation (5)-style calibration:
       The visual evidence bounds the logical danger.
    """

    # ---------------------------------------------------
    # NEW — station disabled during Ablation
    # ---------------------------------------------------
    if not USE_STATION4:
        return 0.0

    """
    Station 4:
    Logical reasoning with GPT — calibrated.
    """

    # -------------------- GPT CALL -----------------------
    client = build_openai_client()

    scores_str = "\n".join(
        f"- {name}: {score:.2f}" for name, score in model_scores.items()
    )

    prompt = f"""
You are an expert in urban traffic safety.

Determine ONLY the logical danger of the anomaly.

ANOMALY:
- type: {anomaly_label}
- description: {anomaly_text}

DETECTOR SCORES:
{scores_str}

YOLO OBJECTS: {objects_summary}

Return STRICT JSON:
{{
  "logic_score": float between 0 and 1
}}
"""

    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    content = response.choices[0].message.content.strip()

    # ----------------- Parse JSON safely -----------------
    try:
        data = json.loads(content)
        L_raw = float(data["logic_score"])
        L_raw = float(np.clip(L_raw, 0, 1))
    except Exception:
        L_raw = extract_number_fallback(content)

    # -----------------------------------------------------
    # STEP 2: Visual confidence C_v
    # -----------------------------------------------------
    yolo_clip = model_scores.get("yolo_clip", 0.0)
    sam_clip = model_scores.get("sam_clip", 0.0)
    dino_score = model_scores.get("dino", 0.0)

    C_v = max(yolo_clip, sam_clip, dino_score)
    C_v = float(np.clip(C_v, 0.0, 1.0))

    # -----------------------------------------------------
    # STEP 3: Final calibrated logic score
    # -----------------------------------------------------
    L_final = C_v * L_raw
    L_final = float(np.clip(L_final, 0.0, 1.0))

    return L_final
