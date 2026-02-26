#!/usr/bin/env python3
import os
import json
import base64
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import cv2
import numpy as np
from PIL import Image, ImageOps
from openai import OpenAI



# ============================
# 1) EXACT prompt template (same as your zero-shot)
# ============================
PROMPT_TEMPLATE = """You are an expert traffic and urban safety analyst and computer vision assistant.

You are given a single FRAME from an urban driving scene or sidewalk.

CLIP INFO (may be empty):
{clip_info}

Your tasks:
1) Decide if the scene is from road or sidewalk (choose one).
2) Provide a concise scene summary (1-2 sentences).
3) Identify relevant objects that might affect safety (cars, bikes, pedestrians, traffic lights, signs, obstacles, roadworks, etc.).
4) Identify anomalies (unusual/abnormal safety-relevant situations).
5) For each anomaly:
   a) Provide a risk score 1-10 (integer).
   b) Provide risk target(s): pedestrians / drivers / cyclists / scooter_riders / workers / general_civilians / emergency_services.
   c) Provide REGION:
      - ALWAYS provide a bounding box in normalized center format:
        x_center_norm, y_center_norm, width_norm, height_norm (all in [0,1]).
      - IF POSSIBLE, ALSO provide an object-shaped polygon outline:
        polygon_norm = list of [x_norm, y_norm] points (>= 8 points, <= 80), closed by the renderer.
        Use polygon_norm when the anomaly is not well-captured by a rectangle (e.g., puddles, cracks, irregular debris).
6) Provide concise reasoning for each anomaly that refers only to visible evidence.
CRITICAL: Segmentation quality matters. For each anomaly you MUST output:
(A) a TIGHT bounding box around ONLY the anomaly object(s)
(B) a polygon outline tracing the anomaly boundary (8–40 vertices)
(C) point prompts for segmentation: positive_points and negative_points

Rules:
- Do NOT segment shadows, lighting patterns, or normal texture variation.
- If the anomaly is “road cracks” or “surface damage”: polygon must trace ONLY the crack/damage pixels, not the whole pavement.
- If the anomaly is “sidewalk obstruction”: polygon must trace ONLY the obstructing items (branches, box, debris), not the sidewalk surface.
- Box must be the smallest rectangle that still fully contains the polygon.
- Points:
  - positive_points: inside anomaly, spread across it
  - negative_points: on nearby background that is visually similar (pavement, curb, wall), close to the boundary
- If there are multiple disconnected anomaly parts: provide multiple polygons (poly_list) and multiple point sets.

Before finalizing each anomaly, perform a “self-check”:
- Ask: “Would my polygon include background pavement?” If yes, shrink it.
- Ask: “Would my polygon miss key anomaly pixels?” If yes, expand slightly.

Return STRICT JSON with the following schema (no extra keys):
{
  "scene_type": "road" | "sidewalk",
  "scene_summary": "string",
  "objects": [
    {
      "id": "obj_1",
      "class": "string",
      "description": "string",
      "position": "string"
    }
  ],
  "anomalies": [
    {
      "id": "anom_1",
      "related_objects": ["obj_1"],
      "type": "string",
      "description": "string",
      "risk_score_1_to_10": 1,
      "risk_target": ["pedestrians"],
      "region": {
        "box": { "x1": 0.0, "y1": 0.0, "x2": 0.0, "y2": 0.0 },
        "poly": [ [0.0,0.0], [0.0,0.0] ],
        "positive_points": [ [0.0,0.0] ],
        "negative_points": [ [0.0,0.0] ]
      },
      "reasoning": "string"
    }
  ],
  "overall_risk_score_1_to_10": 1
}
"""

# ============================
# 2) Image helpers (EXIF-safe) -> base64 for GPT-Vision
# ============================
def exif_corrected_image_bytes(path: Path, max_side: Optional[int] = None) -> Tuple[bytes, int, int]:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")

    if max_side is not None:
        w, h = img.size
        scale = max(w, h) / float(max_side)
        if scale > 1.0:
            new_w = int(round(w / scale))
            new_h = int(round(h / scale))
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    w, h = img.size
    from io import BytesIO
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue(), w, h






# ============================
# 6) GPT-VISION call (image + SAME prompt template)
# ============================
def run_gpt_vision(image_path: Path, model_name: str = "gpt-5.1"):
    img_bytes, _, _ = exif_corrected_image_bytes(image_path, max_side=None)
    b64 = base64.b64encode(img_bytes).decode("utf-8")

    prompt = PROMPT_TEMPLATE.replace("{clip_info}", "")



    messages = [
        {"role": "system", "content": "You are a careful vision-and-safety analyst. Return STRICT JSON only."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        },
    ]

    client = OpenAI()
    resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


def wrap_text(text: str, max_width: int, font, font_scale, thickness):
    """
    Splits text into multiple lines so it fits within max_width.
    """
    words = text.split(" ")
    lines = []
    current_line = ""

    for word in words:
        test_line = current_line + (" " if current_line else "") + word
        text_size = cv2.getTextSize(test_line, font, font_scale, thickness)[0]

        if text_size[0] <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    return lines




def draw_header_and_summary(img: np.ndarray, result: dict, dataset_name: str):
    h, w = img.shape[:2]

    scene_summary = result.get("scene_summary", "")
    anomalies = result.get("anomalies", [])
    overall_risk = result.get("overall_risk_score_1_to_10", 0)
    num_anomalies = len(anomalies)

    header_text = f"{dataset_name} | overall_risk={overall_risk} | anomalies={num_anomalies}"

    # ---- FONT SETTINGS ----
    header_font = cv2.FONT_HERSHEY_SIMPLEX
    header_scale = 1.0
    header_thickness = 2

    summary_font = cv2.FONT_HERSHEY_SIMPLEX
    summary_scale = 0.8
    summary_thickness = 2

    line_spacing = 8
    padding = 15
    max_text_width = w - 2 * padding

    # ---- wrap summary ----
    summary_lines = wrap_text(
        scene_summary,
        max_text_width,
        summary_font,
        summary_scale,
        summary_thickness
    )

    # ---- calculate header height dynamically ----
    header_height = (
        padding * 2 +
        30 +                             # header line
        len(summary_lines) * (28 + line_spacing)
    )

    # ---- draw background ----
    overlay = img.copy()
    cv2.rectangle(
        overlay,
        (0, 0),
        (w, header_height),
        (0, 0, 0),
        thickness=-1
    )
    img[:] = cv2.addWeighted(overlay, 0.75, img, 0.25, 0)

    # ---- draw header text ----
    y = padding + 30
    cv2.putText(
        img,
        header_text,
        (padding, y),
        header_font,
        header_scale,
        (255, 255, 255),
        header_thickness,
        cv2.LINE_AA
    )

    # ---- draw summary lines ----
    y += 20
    for line in summary_lines:
        y += 28
        cv2.putText(
            img,
            line,
            (padding, y),
            summary_font,
            summary_scale,
            (255, 255, 255),
            summary_thickness,
            cv2.LINE_AA
        )


def risk_to_color(risk: int):
    """
    Returns BGR color based on risk score.
    """
    if risk <= 3:
        return (0, 200, 0)      # green
    elif risk <= 6:
        return (0, 255, 255)    # yellow
    elif risk <= 8:
        return (0, 165, 255)    # orange
    else:
        return (0, 0, 255)      # red




# ============================
# 7) VISUALIZATION
# ============================
def visualize_result(image_path: Path, result: dict):
    img = cv2.imread(str(image_path))
    if img is None:
        print("[WARN] Could not load image for visualization.")
        return

    # ---- draw header first ----
    draw_header_and_summary(
        img,
        result,
        dataset_name="input_images"
    )

    h, w = img.shape[:2]

    # We'll draw the risk areas on a separate overlay, then blend into img
    overlay = img.copy()
    alpha = 0.45  # transparency of risk overlay

    for anomaly in result.get("anomalies", []):
        region = anomaly.get("region", {})

        # ---- choose color by risk ----
        risk = int(anomaly.get("risk_score_1_to_10", 0))
        color = risk_to_color(risk)

        # ---- fill risky area by polygon ONLY (no box, no poly line) ----
        poly = region.get("poly", [])
        if poly:
            pts = np.array(
                [[int(x * w), int(y * h)] for x, y in poly],
                dtype=np.int32
            )
            cv2.fillPoly(overlay, [pts], color)

    # ---- blend overlay onto original image ----
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    cv2.namedWindow("Unified Pipeline Output", cv2.WINDOW_NORMAL)
    cv2.imshow("Unified Pipeline Output", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ============================
# 8) TEST: run on a single image
# ============================
if __name__ == "__main__":
    print("[TEST] Starting GPT-only pipeline...")

    BASE_DIR = Path(__file__).resolve().parents[2]  # project root
    test_image = BASE_DIR / "data" / "raw" / "input_image" / "004.png"

    result = run_gpt_vision(test_image)

    print(json.dumps(result, ensure_ascii=False, indent=2))

    visualize_result(test_image, result)




