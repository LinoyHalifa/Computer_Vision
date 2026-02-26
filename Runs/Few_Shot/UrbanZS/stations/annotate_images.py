#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

# Optional high-quality segmentation (SAM). If unavailable, we fall back to GrabCut.
try:
    from segment_anything import sam_model_registry, SamPredictor  # type: ignore
    _SAM_AVAILABLE = True
except Exception:
    _SAM_AVAILABLE = False


def load_results(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def is_success(rec: Dict[str, Any]) -> bool:
    return isinstance(rec.get("result"), dict)


def safe_get_overall_risk(rec: Dict[str, Any]) -> int:
    try:
        return int(rec["result"].get("overall_risk_score_1_to_10", 0))
    except Exception:
        return 0


def risk_to_color(risk: int) -> Tuple[int, int, int]:
    """Map risk 0-10 to a BGR color (OpenCV)."""
    if risk <= 2:
        return (0, 200, 0)       # green
    if risk <= 5:
        return (0, 220, 220)     # yellow-ish
    if risk <= 8:
        return (0, 140, 255)     # orange
    return (0, 0, 255)          # red


def read_image_exif_corrected_bgr(path: Path) -> Optional[np.ndarray]:
    """
    Read image with EXIF orientation applied (important for phone images).
    Returns BGR uint8 image or None.
    """
    try:
        pil = Image.open(path)
        pil = ImageOps.exif_transpose(pil)
        rgb = np.array(pil.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return bgr
    except Exception:
        return None


def _clamp01(x: float) -> float:
    return min(max(x, 0.0), 1.0)


def _norm_box_to_px(
    x_center_norm: float,
    y_center_norm: float,
    width_norm: float,
    height_norm: float,
    img_w: int,
    img_h: int,
) -> Optional[Tuple[int, int, int, int]]:
    x_c = _clamp01(float(x_center_norm))
    y_c = _clamp01(float(y_center_norm))
    w_n = _clamp01(float(width_norm))
    h_n = _clamp01(float(height_norm))

    if w_n <= 0.0 or h_n <= 0.0:
        return None

    w_px = int(round(w_n * img_w))
    h_px = int(round(h_n * img_h))
    cx_px = int(round(x_c * img_w))
    cy_px = int(round(y_c * img_h))

    x1 = max(0, cx_px - w_px // 2)
    y1 = max(0, cy_px - h_px // 2)
    x2 = min(img_w - 1, cx_px + w_px // 2)
    y2 = min(img_h - 1, cy_px + h_px // 2)

    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _polygon_norm_to_px(poly_norm: List[List[float]], img_w: int, img_h: int) -> Optional[np.ndarray]:
    """
    poly_norm: [[x_norm, y_norm], ...] in [0,1]. Returns Nx1x2 int32 array for cv2.fillPoly.
    """
    if not poly_norm or not isinstance(poly_norm, list):
        return None
    pts: List[List[int]] = []
    for p in poly_norm:
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            continue
        try:
            x = int(round(_clamp01(float(p[0])) * (img_w - 1)))
            y = int(round(_clamp01(float(p[1])) * (img_h - 1)))
            pts.append([x, y])
        except Exception:
            continue
    if len(pts) < 3:
        return None
    return np.array(pts, dtype=np.int32).reshape((-1, 1, 2))


def extract_anomaly_regions(rec: Dict[str, Any], img_w: int, img_h: int) -> List[Dict[str, Any]]:
    """
    Extract anomaly regions:
      - box: (x1,y1,x2,y2) in pixels (derived from normalized center/width/height)
      - poly: polygon points in pixels (preferred; enables IoU-quality masks)

    Each returned item:
      {"box": Optional[Tuple[int,int,int,int]], "poly": Optional[np.ndarray], "risk": int, "label": str}
    """
    regions: List[Dict[str, Any]] = []
    anoms = rec.get("result", {}).get("anomalies", [])
    if not isinstance(anoms, list):
        return regions

    for anom in anoms:
        if not isinstance(anom, dict):
            continue

        region = anom.get("region")
        risk = anom.get("risk_score_1_to_10")
        try:
            risk_i = int(risk) if risk is not None else safe_get_overall_risk(rec)
        except Exception:
            risk_i = safe_get_overall_risk(rec)

        label = anom.get("type") or anom.get("description") or "anomaly"
        label = str(label)[:60]

        box: Optional[Tuple[int, int, int, int]] = None
        poly: Optional[np.ndarray] = None

        if isinstance(region, dict):
            # Preferred: polygon_norm (object-shaped)
            poly_norm = region.get("polygon_norm")
            if isinstance(poly_norm, list):
                poly = _polygon_norm_to_px(poly_norm, img_w, img_h)

            # Fallback: normalized box
            box = _norm_box_to_px(
                region.get("x_center_norm", 0.5),
                region.get("y_center_norm", 0.5),
                region.get("width_norm", 0.5),
                region.get("height_norm", 0.5),
                img_w,
                img_h,
            )

        regions.append({"box": box, "poly": poly, "risk": risk_i, "label": label})

    return regions


def polygon_to_mask(img_h: int, img_w: int, poly_pts: np.ndarray) -> np.ndarray:
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly_pts], 1)
    return mask


def load_sam_predictor(checkpoint_path: str, model_type: str = "vit_b", device: str = "cuda") -> Optional["SamPredictor"]:
    """
    Load SAM predictor once and reuse it across images.
    model_type: vit_h | vit_l | vit_b
    device: cuda | cpu
    """
    if not _SAM_AVAILABLE:
        return None

    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        return None

    sam = sam_model_registry[model_type](checkpoint=str(ckpt))
    sam.to(device=device)
    sam.eval()
    return SamPredictor(sam)


def sam_segment_box(predictor: "SamPredictor", img_bgr: np.ndarray, box_xyxy: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    """
    Segment using SAM with a bounding-box prompt.
    Returns mask (uint8 0/1) or None if it fails.
    """
    if predictor is None:
        return None

    x1, y1, x2, y2 = box_xyxy
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(img_rgb)

    input_box = np.array([x1, y1, x2, y2], dtype=np.float32)
    masks, scores, _ = predictor.predict(box=input_box[None, :], multimask_output=True)

    if masks is None or len(masks) == 0:
        return None

    best = int(np.argmax(scores))
    return masks[best].astype(np.uint8)


def grabcut_segment(img_bgr: np.ndarray, box: Tuple[int, int, int, int], iter_count: int = 6) -> np.ndarray:
    """
    GrabCut initialized with rectangle + post-filtering.
    Returns mask uint8 with 0/1.
    """
    x1, y1, x2, y2 = box
    h, w = img_bgr.shape[:2]

    x1 = int(max(0, min(w - 1, x1)))
    x2 = int(max(0, min(w - 1, x2)))
    y1 = int(max(0, min(h - 1, y1)))
    y2 = int(max(0, min(h - 1, y2)))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((h, w), np.uint8)

    rect = (x1, y1, max(1, x2 - x1), max(1, y2 - y1))

    mask = np.zeros((h, w), np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(img_bgr, mask, rect, bgdModel, fgdModel, iter_count, cv2.GC_INIT_WITH_RECT)
        seg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    except Exception:
        seg = np.zeros((h, w), np.uint8)
        seg[y1:y2, x1:x2] = 1
        return seg

    # Restrict to the rectangle area (prevents global spill)
    rect_only = np.zeros_like(seg)
    rect_only[y1:y2, x1:x2] = 1
    seg = (seg & rect_only).astype(np.uint8)

    # Keep the connected component closest to box center (reduces "entire sidewalk" masks)
    if seg.sum() > 0:
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(seg, connectivity=8)
        if num > 1:
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            best = None
            best_score = None
            for i in range(1, num):
                area = stats[i, cv2.CC_STAT_AREA]
                if area < 100:
                    continue
                cxi, cyi = centroids[i]
                dist = ((cxi - cx) ** 2 + (cyi - cy) ** 2) ** 0.5
                score = dist / max(1.0, area ** 0.5)
                if best is None or score < best_score:
                    best = i
                    best_score = score
            if best is not None:
                seg = (labels == best).astype(np.uint8)

    # If mask is extremely tiny, fall back to the rectangle (useful for puddles/cracks).
    box_area = max(1, (x2 - x1) * (y2 - y1))
    area_ratio = float(seg.sum()) / float(box_area)
    if area_ratio < 0.02:
        seg = np.zeros((h, w), np.uint8)
        seg[y1:y2, x1:x2] = 1

    return seg


def highlight_region(img_bgr: np.ndarray, mask01: np.ndarray, color_bgr: Tuple[int, int, int], alpha: float = 0.70) -> np.ndarray:
    """
    Overlay color on mask. alpha=0.70 keeps texture visible (important for cracks/puddles).
    """
    out = img_bgr.copy()
    if mask01 is None or mask01.sum() == 0:
        return out

    overlay = np.zeros_like(out)
    overlay[:, :] = color_bgr

    m = mask01.astype(bool)
    out[m] = cv2.addWeighted(out[m], 1.0 - alpha, overlay[m], alpha, 0)
    return out


def wrap_text(text: str, max_chars: int) -> List[str]:
    words = text.split()
    lines: List[str] = []
    cur: List[str] = []
    cur_len = 0
    for w in words:
        extra = len(w) + (1 if cur else 0)
        if cur_len + extra > max_chars:
            if cur:
                lines.append(" ".join(cur))
            cur = [w]
            cur_len = len(w)
        else:
            cur.append(w)
            cur_len += extra
    if cur:
        lines.append(" ".join(cur))
    return lines


def add_global_summary(img_bgr: np.ndarray, dataset: str, overall_risk: int, num_anoms: int, scene_summary: str) -> np.ndarray:
    lines = [f"{dataset} | overall_risk={overall_risk} | anomalies={num_anoms}"]
    if scene_summary:
        scene_summary = scene_summary[:240]
        lines.extend(wrap_text(scene_summary, max_chars=80))

    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w = img_bgr.shape[:2]
    font_scale = max(0.6, w / 1500.0)
    thickness = 2
    line_h = int(26 * font_scale) + 8

    margin_x, margin_y = 10, 10
    max_tw = 0
    for line in lines:
        (tw, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
        max_tw = max(max_tw, tw)
    rect_h = line_h * len(lines) + 10

    cv2.rectangle(
        img_bgr,
        (margin_x - 6, margin_y - 6),
        (margin_x + max_tw + 10, margin_y + rect_h),
        (0, 0, 0),
        -1,
    )

    y = margin_y + int(22 * font_scale)
    for line in lines:
        cv2.putText(img_bgr, line, (margin_x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y += line_h

    return img_bgr


def annotate_image(img_bgr: np.ndarray, rec: Dict[str, Any], sam_predictor: Optional["SamPredictor"]) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    dataset = rec.get("dataset", "UNKNOWN")
    overall_risk = safe_get_overall_risk(rec)
    anomalies = rec.get("result", {}).get("anomalies", [])
    num_anoms = len(anomalies) if isinstance(anomalies, list) else 0
    scene_summary = rec.get("result", {}).get("scene_summary", "")

    regions = extract_anomaly_regions(rec, w, h)

    # IMPORTANT: segment from the original image (img_bgr), not the progressively-overlaid output.
    out = img_bgr.copy()

    for region in regions:
        risk = int(region["risk"])
        label = str(region["label"])
        color = risk_to_color(risk)

        mask01: Optional[np.ndarray] = None

        if region.get("poly") is not None:
            mask01 = polygon_to_mask(h, w, region["poly"])
        elif region.get("box") is not None:
            box = region["box"]

            # Prefer SAM if available and loaded; otherwise GrabCut fallback
            if sam_predictor is not None:
                mask01 = sam_segment_box(sam_predictor, img_bgr, box)
            if mask01 is None:
                mask01 = grabcut_segment(img_bgr, box, iter_count=6)

        if mask01 is None or int(mask01.sum()) == 0:
            continue

        out = highlight_region(out, mask01, color, alpha=0.70)

        # Outline: polygon preferred; else bbox.
        if region.get("poly") is not None:
            cv2.polylines(out, [region["poly"]], isClosed=True, color=color, thickness=2)
        elif region.get("box") is not None:
            x1, y1, x2, y2 = region["box"]
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # Label placement: top-left of box if present; otherwise centroid of mask.
        text = f"risk={risk} {label}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.55, min(w, h) / 1400.0)
        thickness = 2
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)

        if region.get("box") is not None:
            x1, y1, _, _ = region["box"]
            tx, ty = x1, max(0, y1 - 6)
        else:
            ys, xs = np.where(mask01 == 1)
            tx = int(xs.mean()) if len(xs) else 10
            ty = int(ys.mean()) if len(ys) else 30

        cv2.rectangle(out, (tx, max(0, ty - th - 6)), (tx + tw + 6, ty), (0, 0, 0), -1)
        cv2.putText(out, text, (tx + 3, ty - 3), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    out = add_global_summary(out, dataset, overall_risk, num_anoms, scene_summary)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate images with anomaly masks (polygon preferred; SAM/GrabCut fallback)."
    )
    parser.add_argument("--results", required=True, help="Path to results JSONL produced by run_llm_eval.py")
    parser.add_argument("--output-dir", required=True, help="Output directory for annotated images")
    parser.add_argument("--root-dir", default=".", help="Root dir prepended to frame_path if relative")
    parser.add_argument("--max-images", type=int, default=200, help="Max images to annotate")

    parser.add_argument(
        "--sam-checkpoint",
        default="",
        help="Path to SAM checkpoint (.pth). If provided and SAM installed, SAM will be used.",
    )
    parser.add_argument(
        "--sam-model-type",
        default="vit_b",
        choices=["vit_h", "vit_l", "vit_b"],
        help="SAM model type.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for SAM (cuda recommended).",
    )

    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    sam_predictor = None
    if args.sam_checkpoint:
        sam_predictor = load_sam_predictor(
            checkpoint_path=args.sam_checkpoint,
            model_type=args.sam_model_type,
            device=args.device,
        )
        if sam_predictor is None:
            print("[WARN] SAM requested but not available; falling back to GrabCut.")
        else:
            print(f"[INFO] Using SAM for segmentation: {args.sam_model_type} on {args.device}")

    records = load_results(results_path)
    successes = [r for r in records if is_success(r)]
    print(f"[INFO] Total records: {len(records)}")
    print(f"[INFO] Successful parses: {len(successes)}")

    count = 0
    for rec in successes:
        if count >= args.max_images:
            break

        frame_path = rec.get("frame_path")
        if not frame_path:
            continue

        frame_full = (Path(args.root_dir) / frame_path).resolve()
        if not frame_full.exists():
            print(f"[WARN] Frame not found: {frame_full}")
            continue

        img = read_image_exif_corrected_bgr(frame_full)
        if img is None:
            print(f"[WARN] Failed to read: {frame_full}")
            continue

        annotated = annotate_image(img, rec, sam_predictor)

        ds = rec.get("dataset", "UNKNOWN")
        out_dir = out_root / ds
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / Path(frame_path).name
        cv2.imwrite(str(out_path), annotated)

        count += 1
        print(f"[INFO] Saved: {out_path}")

    print(f"[INFO] Done. Annotated {count} images to {out_root}")


if __name__ == "__main__":
    main()
