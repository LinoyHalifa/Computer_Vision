#!/usr/bin/env python3
import os
import json
import base64
import argparse
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from PIL import Image, ImageOps

from openai import OpenAI


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
{{
  "scene_type": "road" | "sidewalk",
  "scene_summary": "string",
  "objects": [
    {{
      "id": "obj_1",
      "class": "string",
      "description": "string",
      "position": "string"
    }}
  ],
  "anomalies": [
    {{
      "id": "anom_1",
      "related_objects": ["obj_1"],
      "type": "string",
      "description": "string",
      "risk_score_1_to_10": 1,
      "risk_target": ["pedestrians"],
      "region": {{
  "box": {{ "x1": 0.0, "y1": 0.0, "x2": 0.0, "y2": 0.0 }},
  "poly": [ [0.0,0.0], [0.0,0.0] ],
  "positive_points": [ [0.0,0.0] ],
  "negative_points": [ [0.0,0.0] ]
}},
"reasoning": "string"

    }}
  ],
  "overall_risk_score_1_to_10": 1
}}
"""


def exif_corrected_image_bytes(path: Path, max_side: Optional[int] = None) -> Tuple[bytes, int, int]:
    """
    Load image, apply EXIF orientation, optionally downscale, return JPEG bytes + (w,h).
    Using the same EXIF-corrected image for BOTH model input and annotation avoids coordinate mismatch.
    """
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


def build_messages(rec: Dict[str, Any], root_dir: str) -> Tuple[str, List[Dict[str, Any]]]:
    sample_id = rec.get("sample_id") or rec.get("id") or ""
    frame_path = rec.get("frame_path") or rec.get("path") or rec.get("image_path") or ""
    clip_info = rec.get("clip_info") or ""

    if not frame_path:
        raise ValueError("Record missing frame_path/image_path")

    frame_full = (Path(root_dir) / frame_path).resolve()
    if not frame_full.exists():
        raise FileNotFoundError(f"Frame not found: {frame_full}")

    img_bytes, _, _ = exif_corrected_image_bytes(frame_full, max_side=None)
    b64 = base64.b64encode(img_bytes).decode("utf-8")

    prompt = PROMPT_TEMPLATE.format(clip_info=clip_info)

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
    return sample_id, messages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Input manifest JSONL")
    parser.add_argument("--output", required=True, help="Output results JSONL")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--model", default="gpt-5.1")
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)

    manifest_path = Path(args.manifest)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    print(f"[INFO] Loaded {len(records)} records from manifest.")
    if args.shuffle:
        random.shuffle(records)
    if args.limit is not None:
        records = records[: args.limit]

    print(f"[INFO] Evaluating {len(records)} samples.")

    client = OpenAI()

    with out_path.open("w", encoding="utf-8") as out_f:
        for i, rec in enumerate(records, 1):
            try:
                sample_id, messages = build_messages(rec, root_dir=args.root_dir)
                resp = client.chat.completions.create(
                    model=args.model,
                    messages=messages,
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                content = resp.choices[0].message.content

                print("\n===== RAW MODEL OUTPUT =====")
                print(content)
                print("===========================\n")

                result = json.loads(content)

                row = {
                    "sample_id": rec.get("sample_id") or rec.get("id") or sample_id,
                    "dataset": rec.get("dataset", ""),
                    "frame_path": rec.get("frame_path") or rec.get("image_path") or "",
                    "clip_path": rec.get("clip_path", ""),
                    "model": args.model,
                    "result": result,
                }
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"[INFO] {i}/{len(records)} OK: {row['sample_id']}")
            except Exception as e:
                row = {
                    "sample_id": rec.get("sample_id") or rec.get("id") or "",
                    "dataset": rec.get("dataset", ""),
                    "frame_path": rec.get("frame_path") or rec.get("image_path") or "",
                    "clip_path": rec.get("clip_path", ""),
                    "model": args.model,
                    "error": str(e),
                }
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"[WARN] {i}/{len(records)} FAIL: {row['sample_id']} :: {e}")

    print(f"[INFO] Wrote results to: {out_path}")


if __name__ == "__main__":
    main()
