import json

from utils.common import GPT_MODEL
from utils.loaders import build_openai_client


SEVERITY_RANGES = {
    # High danger
    "pedestrian_crossing_on_red_light": (8, 10),
    "foreign_object_on_road": (6, 10),
    "animal_on_road": (6, 10),


    # Medium–High
    "vehicle_on_sidewalk": (5, 9),
    "large_obstacle_on_sidewalk": (5, 9),
    "pothole": (4, 8),

    # Medium
    "crack_on_sidewalk": (3, 6),
    "damaged_sidewalk": (3, 6),
    "vegetation_on_road": (3, 6),

    # Low–Medium
    "puddle_on_sidewalk": (1, 4),
    "illegal_parking": (2, 5),

    # No danger
    "normal_scene": (0, 0),
    "person_walking_on_sidewalk": (0, 0),
    "normal_sidewalk_scene": (0, 0),
    "normal_scooter_parking": (0, 0),
    "normal_empty_sidewalk": (0, 0),
    "person_walking_on_sidewalk_with_a_dog": (0, 0),
    "normal_sidewalk_with_parked_cars_and_streetlight": (0, 0),
    "scooter_park_correctly": (0, 0),
    "normal_active_intersection": (0, 0),
}


def gpt_explain_anomaly(anomaly_label, anomaly_text, severity, model_scores, objects_summary):
    """
    Final GPT step:
    - What is detected (label)
    - Description (from your text)
    - Risk sentence (from GPT)
    - Severity 0-10 (from GPT, clipped to your range)
    """

    # טווח severity לפי לייבל
    min_s, max_s = SEVERITY_RANGES.get(anomaly_label, (0, 10))

    # ===== 1. danger map (your logic) =====
    danger_map = {
        "obstacle_on_sidewalk": ["pedestrians"],
        "vehicle_on_sidewalk": ["pedestrians"],
        "pedestrian_on_road": ["drivers"],
        "animal_on_road": ["drivers"],
        "pothole": ["drivers"],
        "foreign_object_on_road": ["drivers"],
        "vegetation_on_sidewalk": ["pedestrians"],
        "vegetation_on_road": ["drivers"],
        "normal_scene": [],
        "large_obstacle_on_sidewalk": ["pedestrians"],
        "puddle_on_sidewalk": ["pedestrians"],
        "illegal_parking": ["pedestrians"],
        "a_dress_on_road": ["drivers"],
        "crack_on_sidewalk": ["pedestrians"],
        "damaged_sidewalk": ["pedestrians"],
        "pedestrian_crossing_on_red_light": ["pedestrians", "drivers"],

    }

    client = build_openai_client()

    scores_str = "\n".join(
        f"- {name}: {score:.2f}" for name, score in model_scores.items()
    )

    # ===== 2. GPT Prompt =====
    prompt = f"""
You are a traffic safety expert.

Your task:
Return a STRICT JSON object with the following exact structure:

{{
  "what_is_detected": "<label>",
  "description": "<description>",
  "risk": "<short risk sentence or 'no risk'>",
  "severity_0_10": <number>
}}

Mandatory rules:
- what_is_detected MUST be exactly the label: "{anomaly_label}".
- description MUST be exactly: "{anomaly_text}".
- DO NOT rewrite or modify the label.
- DO NOT rewrite or modify the description.
- DO NOT add additional explanation.
- DO NOT describe objects or mention detector scores.
- Only determine risk + severity logically.

IMPORTANT SEVERITY RANGE:
- The severity MUST be a number between {min_s} and {max_s}.
- Never go outside this range.

If the label represents a NORMAL scene (no anomaly):
- risk MUST be "no risk"
- severity MUST be 0

If the label represents an ANOMALY:
- risk MUST briefly state who is endangered.

Input information:
Label: {anomaly_label}
Description: {anomaly_text}
Detected objects: {objects_summary}
Detector scores (0-1):
{scores_str}

Return ONLY valid JSON.
"""

    resp = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    content = resp.choices[0].message.content.strip()

    # ===== 3. Parse JSON safely =====
    try:
        data = json.loads(content)
    except Exception:
        data = {
            "what_is_detected": anomaly_label,
            "description": anomaly_text,
            "risk": "Automatic analysis generated this summary.",
            "severity_0_10": severity * 10.0,
        }

    # לוודא שיש description גם אם GPT פישל
    if "description" not in data or not data["description"]:
        data["description"] = anomaly_text

    # ===== 4. Clip severity לטווח שהגדרת =====
    try:
        sev = float(data.get("severity_0_10", severity * 10.0))
    except Exception:
        sev = severity * 10.0

    sev = max(min(sev, max_s), min_s)
    data["severity_0_10"] = sev

    # ===== 5. להוסיף danger_to מהמפה שלך =====
    true_dangers = danger_map.get(anomaly_label, [])
    data["danger_to"] = true_dangers

    # ===== 5.1 Explicit no danger for normal scenes =====
    if not data["danger_to"]:
        data["danger_to"] = ["There Is No Danger"]

    # ===== 6. למפות את risk של GPT ל-risk_sentence =====
    # כאן אנחנו *לא* דורכים מעל GPT, רק מעבירים לשם השדה שה-GUI שלך מצפה לו
    gpt_risk = data.get("risk", "")
    if anomaly_label == "normal_scene":
        data["risk_sentence"] = "There is no risk in this scene."
    else:
        data["risk_sentence"] = gpt_risk or "The detected condition may pose a risk."

    return data
