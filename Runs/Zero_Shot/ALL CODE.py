import os
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
import clip
from openai import OpenAI

# Optional imports
try:
    import timm
except ImportError:
    timm = None

try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
except ImportError:
    sam_model_registry = None
    SamAutomaticMaskGenerator = None


# ==========================
# CONFIG
# ==========================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

YOLO_WEIGHTS = "yolov8n.pt"
GPT_MODEL = "gpt-4o-mini"

OUTPUT_DIR = r"C:\LogSAD\LogSAD-master\UrbanZH\outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# List of anomaly labels: (id, text description)
ANOMALY_LABELS = [
    (
        "normal_scene",
        "a normal safe urban road scene with intact asphalt, no cracks, no potholes, and no obstacles"
    ),
    (
        "pedestrian_on_road",
        "a pedestrian walking or standing on the road where cars drive"
    ),
    (
        "animal_on_road",
        "an animal standing on the road where vehicles drive"
    ),
    (
        "crack_in_road",
        "visible cracks or broken asphalt on the road surface"
    ),
    (
        "pothole",
        "a pothole or missing asphalt in the road"
    ),
    (
        "foreign_object_on_road",
        "a foreign object or obstacle lying on the road where vehicles drive"
    ),
    (
        "illegal_parking",
        "a vehicle parked in an unusual or illegal position in the road, blocking traffic"
    ),
]

# Folder with "normal" (non-anomalous) road scenes for DINO patch memory
NORMAL_IMAGES_DIR = r"C:\LogSAD\LogSAD-master\UrbanZH\normal_images"

# SAM configuration
USE_SAM = True
SAM_MODEL_TYPE = "vit_h"
SAM_CHECKPOINT = r"C:\LogSAD\sam_vit_h_4b8939.pth"  # change if needed


# ==========================
# LOADERS
# ==========================

def load_yolo():
    """Load YOLO model for object detection."""
    model = YOLO(YOLO_WEIGHTS)
    return model


def load_clip():
    """Load CLIP model + preprocess."""
    model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    return model, preprocess


def precompute_text_embeddings(clip_model):
    """Compute CLIP embeddings for all anomaly label texts."""
    texts = [desc for (_id, desc) in ANOMALY_LABELS]
    tokens = clip.tokenize(texts).to(DEVICE)
    with torch.no_grad():
        text_features = clip_model.encode_text(tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features.cpu().numpy()  # [num_labels, dim]


def load_dino():
    """Load DINOv2 model + transform for patch-level features."""
    if timm is None:
        raise ImportError(
            "timm is not installed. Install it with:\n"
            "  pip install timm\n"
            "inside your environment."
        )

    model_name = "vit_base_patch14_dinov2.lvd142m"
    model = timm.create_model(model_name, pretrained=True)
    model.eval()
    model.to(DEVICE)

    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    transform = T.Compose([
        T.Resize((518, 518), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5)
        ),
    ])
    return model, transform


def load_sam():
    """Load SAM mask generator."""
    if not USE_SAM:
        raise RuntimeError("SAM is required in this pipeline but USE_SAM=False.")

    if sam_model_registry is None or SamAutomaticMaskGenerator is None:
        raise ImportError(
            "segment_anything is not installed. Install it and its dependencies."
        )

    if not os.path.isfile(SAM_CHECKPOINT):
        raise FileNotFoundError(
            f"SAM checkpoint not found at {SAM_CHECKPOINT}. "
            f"Download it and update SAM_CHECKPOINT."
        )

    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(device=DEVICE)
    mask_generator = SamAutomaticMaskGenerator(sam)
    return mask_generator


def build_openai_client():
    """Create OpenAI client (OPENAI_API_KEY must be set)."""
    return OpenAI()


# ==========================
# UTILS
# ==========================

def cosine_sim(a, b):
    """Cosine similarity assuming a and b are L2-normalized."""
    return float(np.dot(a, b))


def crop_region(img_rgb, bbox):
    """Crop region from RGB image given [x1,y1,x2,y2]."""
    x1, y1, x2, y2 = bbox
    x1 = max(int(x1), 0)
    y1 = max(int(y1), 0)
    x2 = min(int(x2), img_rgb.shape[1])
    y2 = min(int(y2), img_rgb.shape[0])
    return img_rgb[y1:y2, x1:x2]


def clip_image_embedding(clip_model, preprocess, img_rgb):
    """Compute CLIP image embedding from numpy RGB image."""
    pil_img = Image.fromarray(img_rgb)
    tensor = preprocess(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        image_features = clip_model.encode_image(tensor)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features.squeeze(0).cpu().numpy()  # [dim]


# ==========================
# DINO PATCH EMBEDDINGS
# ==========================

def dino_patch_embeddings(dino_model, dino_transform, img_rgb):
    """
    Compute DINOv2 patch embeddings for a RGB image.
    Returns array of shape [num_patches, dim].
    """
    from PIL import Image as PILImage

    pil_img = PILImage.fromarray(img_rgb)
    tensor = dino_transform(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        feats = dino_model.forward_features(tensor)
        # For DINOv2 in timm, patch tokens are often in "x_norm_patchtokens"
        if isinstance(feats, dict):
            if "x_norm_patchtokens" in feats:
                patches = feats["x_norm_patchtokens"]  # [B, N, C]
            elif "token_embeddings" in feats:
                patches = feats["token_embeddings"][:, 1:, :]  # skip CLS
            else:
                # Fallback: try first tensor-like entry and drop CLS if present
                first_key = list(feats.keys())[0]
                patches = feats[first_key]
        else:
            patches = feats

    # Expect shape [1, N, C]
    if patches.dim() == 3:
        patches = patches.squeeze(0)  # [N, C]
    else:
        raise RuntimeError(
            f"Unexpected DINO patch shape: {patches.shape}, expected [1, N, C]."
        )

    # L2 normalize each patch
    patches = patches / patches.norm(dim=-1, keepdim=True)
    return patches.cpu().numpy()  # [num_patches, dim]


def precompute_dino_normal_patch_bank(dino_model, dino_transform):
    """
    Build a memory bank of patch embeddings from NORMAL_IMAGES_DIR.
    Returns array [N_total_patches, dim].
    """
    if not NORMAL_IMAGES_DIR or not os.path.isdir(NORMAL_IMAGES_DIR):
        raise FileNotFoundError(
            f"NORMAL_IMAGES_DIR '{NORMAL_IMAGES_DIR}' does not exist. "
            f"Create it and put normal road images inside."
        )

    all_patches = []

    for fname in os.listdir(NORMAL_IMAGES_DIR):
        fpath = os.path.join(NORMAL_IMAGES_DIR, fname)
        if not os.path.isfile(fpath):
            continue
        img = cv2.imread(fpath)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        patches = dino_patch_embeddings(dino_model, dino_transform, img_rgb)  # [P, C]
        all_patches.append(patches)

    if not all_patches:
        raise RuntimeError("No valid normal images found for DINO patch bank.")

    normal_patch_bank = np.vstack(all_patches)  # [N_total_patches, dim]
    return normal_patch_bank


# ==========================
# STATION 1: YOLO + CLIP
# ==========================

def station_yolo_clip(img_rgb, yolo_results, clip_model, preprocess, text_embs):
    """
    Station 1:
    - Use YOLO detections as regions.
    - For each bbox, compute CLIP image embedding.
    - Compare to all anomaly texts.
    - Return best (label, label_idx, bbox, severity in [0,1]).
    """
    best = {
        "label": "normal_scene",
        "label_idx": 0,
        "clip_sim": -1e9,
        "bbox": None,
    }

    for box, cls, conf in zip(yolo_results.boxes.xyxy,
                              yolo_results.boxes.cls,
                              yolo_results.boxes.conf):
        x1, y1, x2, y2 = box.tolist()
        bbox = [x1, y1, x2, y2]
        crop = crop_region(img_rgb, bbox)
        if crop.shape[0] < 32 or crop.shape[1] < 32:
            continue

        img_emb = clip_image_embedding(clip_model, preprocess, crop)
        sims = text_embs @ img_emb  # [num_labels]
        idx = int(np.argmax(sims))
        sim = float(sims[idx])

        if sim > best["clip_sim"]:
            best["clip_sim"] = sim
            best["label_idx"] = idx
            best["label"] = ANOMALY_LABELS[idx][0]
            best["bbox"] = bbox

    if best["bbox"] is None:
        return None, None, None, 0.0

    # Map CLIP similarity (roughly [-1,1]) to [0,1]
    severity = float(np.clip((best["clip_sim"] + 1.0) / 2.0, 0.0, 1.0))
    return best["label"], best["label_idx"], best["bbox"], severity


# ==========================
# STATION 2: SAM + CLIP
# ==========================

def station_sam_clip(img_rgb, mask_generator, clip_model, preprocess, text_embs):
    """
    Station 2:
    - Use SAM to segment the image.
    - For each mask bounding box, compute CLIP image embedding.
    - Compare to all anomaly texts.
    - Return best (label, label_idx, bbox, severity in [0,1]).
    """
    masks = mask_generator.generate(img_rgb)
    if not masks:
        return None, None, None, 0.0

    H, W, _ = img_rgb.shape

    best = {
        "label": "normal_scene",
        "label_idx": 0,
        "clip_sim": -1e9,
        "bbox": None,
    }

    for m in masks:
        x, y, w, h = m["bbox"]
        x1, y1, x2, y2 = x, y, x + w, y + h
        bbox = [max(0, x1), max(0, y1), min(W, x2), min(H, y2)]
        crop = crop_region(img_rgb, bbox)
        if crop.shape[0] < 32 or crop.shape[1] < 32:
            continue

        img_emb = clip_image_embedding(clip_model, preprocess, crop)
        sims = text_embs @ img_emb
        idx = int(np.argmax(sims))
        sim = float(sims[idx])

        if sim > best["clip_sim"]:
            best["clip_sim"] = sim
            best["label_idx"] = idx
            best["label"] = ANOMALY_LABELS[idx][0]
            best["bbox"] = bbox

    if best["bbox"] is None:
        return None, None, None, 0.0

    severity = float(np.clip((best["clip_sim"] + 1.0) / 2.0, 0.0, 1.0))
    return best["label"], best["label_idx"], best["bbox"], severity


# ==========================
# STATION 3: DINO PATCH-BASED ANOMALY
# ==========================

def station_dino_patches(img_rgb, dino_model, dino_transform, normal_patch_bank):
    """
    Station 3 (A+B):
    - A) Compare each patch in the test image to the normal patch bank.
         For each patch: anomaly_normal = 1 - max_sim_to_normal_patches
         score_dino_normal = max(anomaly_normal)
    - B) Compare each patch to all other patches in the same image (self-similarity).
         For each patch: anomaly_self = 1 - max_sim_to_other_patches
         score_dino_self = max(anomaly_self)
    - Final DINO severity = max(score_dino_normal, score_dino_self) in [0,1].
    """
    # Compute patch embeddings for test image: [P, C]
    patch_embs = dino_patch_embeddings(dino_model, dino_transform, img_rgb)
    if patch_embs.size == 0:
        return 0.0

    # -------- A) Comparison to normal patch bank --------
    # patch_embs: [P, C], normal_patch_bank: [N, C]
    sims_normal = patch_embs @ normal_patch_bank.T  # [P, N]
    # For each test patch, similarity to its most similar normal patch
    patch_sim_normal = sims_normal.max(axis=1)  # [P]
    patch_anomaly_normal = 1.0 - patch_sim_normal
    score_dino_normal = float(np.clip(patch_anomaly_normal.max(), 0.0, 1.0))

    # -------- B) Self-similarity inside the same image --------
    sims_self = patch_embs @ patch_embs.T  # [P, P]
    # Ignore self-similarity on diagonal by setting it to -inf
    np.fill_diagonal(sims_self, -1.0)
    patch_sim_self = sims_self.max(axis=1)  # best similarity to any other patch
    patch_anomaly_self = 1.0 - patch_sim_self
    score_dino_self = float(np.clip(patch_anomaly_self.max(), 0.0, 1.0))

    # Final DINO severity = max over both
    score_dino = max(score_dino_normal, score_dino_self)
    score_dino = float(np.clip(score_dino, 0.0, 1.0))

    return score_dino


# ==========================
# STATION 4: GPT LOGIC SCORING
# ==========================

def gpt_logic_score(anomaly_label, anomaly_text, model_scores, objects_summary):
    """
    Station 4:
    - Use GPT to reason about logical risk based on:
      * anomaly type
      * scores from previous stations
      * objects detected by YOLO
    - Returns logic_score in [0,1].
    """
    client = build_openai_client()

    scores_str = "\n".join(
        f"- {name}: {score:.2f}" for name, score in model_scores.items()
    )

    prompt = f"""
You are a traffic safety expert.

We have an urban road scene.

Anomaly (candidate):
- id: {anomaly_label}
- description: {anomaly_text}

Detector scores (0-1):
{scores_str}

Objects detected by YOLO in the scene: {objects_summary}

Based on the anomaly type, scores, and objects, estimate how severe the logical violation is
(for example: pedestrian on road with cars is more severe than a small pothole with no cars).

Return ONLY a JSON object with the following key:
- "logic_score": a single float in [0,1] (0 = no logical issue, 1 = very dangerous logical situation)
"""

    resp = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    content = resp.choices[0].message.content.strip()
    try:
        data = json.loads(content)
        logic_score = float(data.get("logic_score", 0.0))
        logic_score = float(np.clip(logic_score, 0.0, 1.0))
    except Exception:
        logic_score = 0.0

    return logic_score


# ==========================
# FINAL GPT EXPLANATION
# ==========================

def gpt_explain_anomaly(anomaly_label, anomaly_text, severity, model_scores, objects_summary):
    """
    Final GPT step:
    - Explain what is detected
    - What is the risk
    - Who is in danger
    - Severity 0-10 (aligned with the numeric severity)
    """
    client = build_openai_client()

    scores_str = "\n".join(
        f"- {name}: {score:.2f}" for name, score in model_scores.items()
    )

    prompt = f"""
You are a traffic safety expert.

Anomaly:
- id: {anomaly_label}
- description: {anomaly_text}

Global severity (0-1): {severity:.2f}

Detector scores (0-1):
{scores_str}

Objects detected in the scene: {objects_summary}

Explain your answer in STRICT JSON with the following keys:
- "what_is_detected": short natural language description of the anomaly in the scene
- "risk_sentence": one sentence describing the main risk
- "danger_to": list of groups at risk (choose from: drivers, pedestrians, cyclists, motorcyclists, children, elderly)
- "severity_0_10": a float between 0 and 10 that reflects the overall severity
"""

    resp = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    content = resp.choices[0].message.content.strip()
    try:
        data = json.loads(content)
    except Exception:
        data = {
            "what_is_detected": f"An anomaly of type {anomaly_label}.",
            "risk_sentence": f"Potential risk related to {anomaly_label}.",
            "danger_to": ["drivers"],
            "severity_0_10": severity * 10.0
        }

    return data


# ==========================
# VISUALIZATION
# ==========================

def save_annotated(image_path, img_bgr, bbox, label, severity_0_10=None):
    """Save annotated image with bbox and label."""
    img_vis = img_bgr.copy()

    if bbox is not None:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

        text = label if label is not None else ""
        if severity_0_10 is not None:
            text += f" ({severity_0_10:.1f}/10)"

        cv2.putText(
            img_vis,
            text,
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    out_name = Path(image_path).stem + "_annotated.png"
    out_path = str(Path(OUTPUT_DIR) / out_name)
    cv2.imwrite(out_path, img_vis)
    return out_path


# ==========================
# MAIN PIPELINE
# ==========================

def analyze_image(image_path, yolo_model, clip_model, preprocess, text_embs,
                  dino_model, dino_transform, dino_normal_patch_bank,
                  sam_mask_generator):
    """
    Full 4-station pipeline on a single image:
    1) YOLO + CLIP
    2) SAM + CLIP
    3) DINO patch-based anomaly (normal + self)
    4) GPT logic scoring
    Then final GPT explanation.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # YOLO detections (used for station 1 and object summary)
    yolo_results = yolo_model(img_bgr)[0]
    names = yolo_model.names
    yolo_objects = []
    for box, cls, conf in zip(yolo_results.boxes.xyxy,
                              yolo_results.boxes.cls,
                              yolo_results.boxes.conf):
        label = names[int(cls.item())]
        yolo_objects.append(label)
    objects_summary = ", ".join(yolo_objects) if yolo_objects else "no objects detected"

    # === Station 1: YOLO + CLIP ===
    label_yolo, idx_yolo, bbox_yolo, score_yolo_clip = station_yolo_clip(
        img_rgb, yolo_results, clip_model, preprocess, text_embs
    )

    # === Station 2: SAM + CLIP ===
    label_sam, idx_sam, bbox_sam, score_sam_clip = station_sam_clip(
        img_rgb, sam_mask_generator, clip_model, preprocess, text_embs
    )

    # Decide anomaly label + bbox based on highest CLIP-based severity
    label_final = None
    idx_final = None
    bbox_final = None

    if score_yolo_clip >= score_sam_clip and label_yolo is not None:
        label_final = label_yolo
        idx_final = idx_yolo
        bbox_final = bbox_yolo
    elif label_sam is not None:
        label_final = label_sam
        idx_final = idx_sam
        bbox_final = bbox_sam

    # If still nothing or normal_scene → treat as no anomaly
    if label_final is None or label_final == "normal_scene":
        annotated_path = save_annotated(image_path, img_bgr, None, None)
        result = {
            "anomaly_detected": False,
            "anomaly_label": "none",
            "severity_score": 0.0,
            "severity_0_10": 0.0,
            "gpt_explanation": {
                "what_is_detected": "No anomaly detected.",
                "risk_sentence": "Scene appears normal.",
                "danger_to": [],
                "severity_0_10": 0.0
            },
            "bbox": None,
            "objects": yolo_objects,
            "model_scores": {
                "yolo_clip": score_yolo_clip,
                "sam_clip": score_sam_clip,
                "dino": 0.0,
                "gpt_logic": 0.0
            },
            "annotated_image": annotated_path
        }
        return result

    # === Station 3: DINO patch-based anomaly ===
    score_dino = station_dino_patches(
        img_rgb, dino_model, dino_transform, dino_normal_patch_bank
    )

    # Prepare partial scores before GPT logic
    model_scores_partial = {
        "yolo_clip": score_yolo_clip,
        "sam_clip": score_sam_clip,
        "dino": score_dino,
    }

    # === Station 4: GPT logic scoring ===
    _, anomaly_text = ANOMALY_LABELS[idx_final]
    score_gpt_logic = gpt_logic_score(
        anomaly_label=label_final,
        anomaly_text=anomaly_text,
        model_scores=model_scores_partial,
        objects_summary=objects_summary
    )

    # Collect all four scores
    model_scores = {
        "yolo_clip": score_yolo_clip,
        "sam_clip": score_sam_clip,
        "dino": score_dino,
        "gpt_logic": score_gpt_logic,
    }

    # Global severity = max of the four scores
    severity = max(model_scores.values())
    severity_0_10 = float(severity * 10.0)

    # === Final GPT explanation ===
    gpt_info = gpt_explain_anomaly(
        anomaly_label=label_final,
        anomaly_text=anomaly_text,
        severity=severity,
        model_scores=model_scores,
        objects_summary=objects_summary
    )

    # === Save annotated image ===
    annotated_path = save_annotated(
        image_path, img_bgr, bbox_final, label_final, severity_0_10
    )

    result = {
        "anomaly_detected": True,
            "anomaly_label": label_final,
            "severity_score": severity,
            "severity_0_10": severity_0_10,
            "gpt_explanation": gpt_info,
            "bbox": bbox_final,
            "objects": yolo_objects,
            "model_scores": model_scores,
            "annotated_image": annotated_path
    }

    return result


# ==========================
# MAIN
# ==========================

if __name__ == "__main__":
    IMAGE_PATH = r"C:\LogSAD\LogSAD-master\UrbanZH\Images\000.png"

    print("[INFO] Loading YOLO...")
    yolo_model = load_yolo()

    print("[INFO] Loading CLIP...")
    clip_model, preprocess = load_clip()

    print("[INFO] Precomputing CLIP text embeddings...")
    text_embs = precompute_text_embeddings(clip_model)

    print("[INFO] Loading DINOv2...")
    dino_model, dino_transform = load_dino()

    print("[INFO] Building DINO normal patch bank...")
    dino_normal_patch_bank = precompute_dino_normal_patch_bank(dino_model, dino_transform)

    print("[INFO] Loading SAM...")
    sam_mask_generator = load_sam()

    print(f"[INFO] Analyzing image: {IMAGE_PATH}")
    result = analyze_image(
        IMAGE_PATH,
        yolo_model,
        clip_model,
        preprocess,
        text_embs,
        dino_model=dino_model,
        dino_transform=dino_transform,
        dino_normal_patch_bank=dino_normal_patch_bank,
        sam_mask_generator=sam_mask_generator
    )

    print("\n===== ANOMALY RESULT =====")
    print(json.dumps(result, indent=4, ensure_ascii=False))
