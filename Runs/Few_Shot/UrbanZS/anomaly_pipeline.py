import json
from pathlib import Path

import cv2
import numpy as np

from utils.common import NORMAL_LABELS, ANOMALY_LABELS
from utils.loaders import load_clip, load_dino
from stations.station3_dino import clip_semantic_matching_all_labels


from stations.station3_dino import (
    station_dino_patches,
    precompute_dino_normal_patch_bank
)

with open("statistics.json", "r") as f:
    STATS = json.load(f)



from stations.gpt_explainer import gpt_explain_anomaly


# ======================================================
# PATHS
# ======================================================

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input_images"
OUTPUT_DIR = BASE_DIR / "outputs" / "json"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ======================================================
# INIT MODELS (ONLY CLIP + DINO)
# ======================================================

def init_models():
    print("[INIT] Loading CLIP...")
    clip_model, clip_preprocess = load_clip()

    print("[INIT] Loading DINO...")
    dino_model, dino_transform = load_dino()

    print("[INIT] Building DINO normal patch bank...")
    normal_patch_bank = precompute_dino_normal_patch_bank(
        dino_model, dino_transform
    )

    return {
        "clip_model": clip_model,
        "clip_preprocess": clip_preprocess,   # ✅ חשוב
        "dino": dino_model,
        "dino_transform": dino_transform,
        "normal_patch_bank": normal_patch_bank,
    }


def calibrate_dino(score):
    mean = STATS["dino"]["mean"]
    std = STATS["dino"]["std"] + 1e-6

    z = (score - mean) / std
    #z = max(z, 0.0)          # רק חריגות מהנורמל
    return 1.0 / (1.0 + np.exp(-z))   # sigmoid



# ======================================================
# ANALYZE IMAGE – STATION 3 ONLY + GPT EXPLAINER
# ======================================================

def analyze_image_station3(image_path, models):
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # -------- Station 3: DINO + CLIP --------
    # -------- Station 3: DINO ONLY --------
    score_dino = station_dino_patches(
        img_rgb,
        models["dino"],
        models["dino_transform"],
        models["normal_patch_bank"],
    )

    final_score = calibrate_dino(score_dino)
    anomaly_detected = final_score > 0.73

    # -------- CLIP semantic label selection --------
    scores, best_label, best_score = clip_semantic_matching_all_labels(
        img_rgb=img_rgb,
        clip_model=models["clip_model"],
        clip_preprocess=models["clip_preprocess"],
        normal_labels=NORMAL_LABELS,
        anomaly_labels=ANOMALY_LABELS
    )

    label_text_map = dict(NORMAL_LABELS + ANOMALY_LABELS)
    selected_label_text = label_text_map[best_label]

    # -------- GPT EXPLAINER (POST-HOC ONLY) --------
    gpt_explanation = gpt_explain_anomaly(
        anomaly_label=best_label,
        anomaly_text=selected_label_text,
        severity=final_score,
        model_scores={
            "dino": score_dino,
            "clip_best_score": best_score
        },
        objects_summary="N/A (no object detection used)"
    )

    # -------- FINAL RESULT --------
    result = {
        "image": image_path.name,
        "final_score": float(final_score),
        "score_dino": float(score_dino),
        "anomaly_detected": bool(anomaly_detected),
        "semantic_label": best_label,
        "gpt_explanation": gpt_explanation
    }

    return result

# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":

    # ---- init models ONCE ----
    models = init_models()

    # ---- run on single image (example) ----
    image_path = INPUT_DIR / "023.png"
    print(f"[INFO] Analyzing image: {image_path}")

    result = analyze_image_station3(
        image_path=image_path,
        models=models
    )

    print("\n===== ANOMALY RESULT =====")
    print(json.dumps(result, indent=4, ensure_ascii=False))

    # ---- save JSON ----
    json_out = OUTPUT_DIR / f"{image_path.stem}.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(f"[INFO] JSON saved to: {json_out}")

    # ============================
    # SHOW IMAGE + EXPLANATION BELOW (RESIZABLE WINDOW)
    # ============================

    annotated = cv2.imread(str(image_path))

    if annotated is not None:

        # ---- Resize image so it always fits initially ----
        max_width = 700
        scale = max_width / annotated.shape[1]
        resized = cv2.resize(annotated, None, fx=scale, fy=scale)

        # ---- Create explanation text ----
        explanation = result["gpt_explanation"]
        raw_text = (
            f"What is detected: {explanation.get('what_is_detected', '')}\n"
            f"Description: {explanation.get('description', '')}\n"
            f"Danger to: {', '.join(explanation.get('danger_to', []))}\n"
            f"Severity (0-10): {explanation.get('severity_0_10', '')}"
        )

        # ---- WORD WRAPPING ----
        import textwrap

        max_chars_per_line = 55
        wrapped_lines = []
        for line in raw_text.split("\n"):
            wrapped_lines.extend(textwrap.wrap(line, width=max_chars_per_line))

        # ---- Visual parameters ----
        left_padding = 25
        top_padding = 30
        line_height = 32
        font_scale = 0.70
        thickness = 2

        # ---- White background for text ----
        text_block_height = len(wrapped_lines) * line_height + top_padding + 20
        white_block = 255 * np.ones(
            (text_block_height, resized.shape[1], 3),
            dtype=np.uint8
        )

        # ---- Draw text ----
        y = top_padding
        for line in wrapped_lines:
            cv2.putText(
                white_block, line, (left_padding, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0, 0, 0), thickness, cv2.LINE_AA
            )
            y += line_height

        # ---- Combine image + explanation ----
        final_display = np.vstack((resized, white_block))

        # ---- Open window ----
        cv2.namedWindow("Analysis Result", cv2.WINDOW_NORMAL)
        cv2.imshow("Analysis Result", final_display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print("[WARN] Could not load image for display.")
