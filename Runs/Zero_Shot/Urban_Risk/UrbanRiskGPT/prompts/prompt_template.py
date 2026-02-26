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
      "region": {
  "box": { "x1":..., "y1":..., "x2":..., "y2":... },          // normalized
  "poly": [ [x,y], [x,y], ... ],                               // normalized
  "positive_points": [ [x,y], [x,y], ... ],                    // normalized
  "negative_points": [ [x,y], [x,y], ... ]                     // normalized},
      "reasoning": "string"
    }}
  ],
  "overall_risk_score_1_to_10": 1
}}
"""
