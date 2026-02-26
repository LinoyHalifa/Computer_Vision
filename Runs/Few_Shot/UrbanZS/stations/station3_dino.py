import os
from pathlib import Path

import cv2
import numpy as np
import torch
import clip

from PIL import Image

from utils.common import USE_STATION3, DEVICE

# ======================================================
# PATHS
# ======================================================

BASE_DIR = Path(__file__).resolve().parent.parent
NORMAL_IMAGES_DIR = BASE_DIR / "normal_images"


# ======================================================
# DINO PATCH EMBEDDINGS
# ======================================================

def dino_patch_embeddings(dino_model, dino_transform, img_rgb: np.ndarray) -> np.ndarray:
    """
    Compute DINOv2 patch embeddings for an RGB image.
    Returns array of shape [num_patches, dim].
    """

    pil_img = Image.fromarray(img_rgb)
    tensor = dino_transform(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        feats = dino_model.forward_features(tensor)

        if isinstance(feats, dict):
            if "x_norm_patchtokens" in feats:
                patches = feats["x_norm_patchtokens"]          # [1, N, C]
            elif "token_embeddings" in feats:
                patches = feats["token_embeddings"][:, 1:, :]  # skip CLS
            else:
                patches = feats[list(feats.keys())[0]]
        else:
            patches = feats

    if patches.dim() != 3:
        raise RuntimeError(f"Unexpected DINO patch shape: {patches.shape}")

    patches = patches.squeeze(0)                               # [N, C]
    patches = patches / patches.norm(dim=-1, keepdim=True)    # L2 normalize

    return patches.cpu().numpy()


# ======================================================
# DINO NORMAL PATCH BANK
# ======================================================

def precompute_dino_normal_patch_bank(dino_model, dino_transform) -> np.ndarray:
    """
    Build a memory bank of patch embeddings from NORMAL_IMAGES_DIR.
    Returns array [N_total_patches, dim].
    """

    if not NORMAL_IMAGES_DIR.is_dir():
        raise FileNotFoundError(f"{NORMAL_IMAGES_DIR} does not exist.")

    all_patches = []

    for fname in os.listdir(NORMAL_IMAGES_DIR):
        fpath = NORMAL_IMAGES_DIR / fname
        if not fpath.is_file():
            continue

        img = cv2.imread(str(fpath))
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        patches = dino_patch_embeddings(dino_model, dino_transform, img_rgb)
        all_patches.append(patches)

    if not all_patches:
        raise RuntimeError("No valid normal images found.")

    return np.vstack(all_patches)


# ======================================================
# DINO ANOMALY SCORE
# ======================================================

def station_dino_patches(img_rgb, dino_model, dino_transform, normal_patch_bank):
    """
    Patch-level anomaly detection with DINO.
    """

    if not USE_STATION3:
        return 0.0

    patch_embs = dino_patch_embeddings(dino_model, dino_transform, img_rgb)
    if patch_embs.size == 0:
        return 0.0

    # A) Normal memory bank comparison
    sims_normal = patch_embs @ normal_patch_bank.T
    patch_anomaly_normal = 1.0 - sims_normal.max(axis=1)
    score_dino_normal = patch_anomaly_normal.max()

    # B) Self-similarity
    sims_self = patch_embs @ patch_embs.T
    np.fill_diagonal(sims_self, -1.0)
    patch_anomaly_self = 1.0 - sims_self.max(axis=1)
    score_dino_self = patch_anomaly_self.max()

    return float(np.clip(max(score_dino_normal, score_dino_self), 0.0, 1.0))


# ======================================================
# CLIP IMAGE ↔ TEXT (PAIRWISE)
# ======================================================

def clip_image_text_similarity(clip_model, preprocess, img_rgb: np.ndarray, text: str) -> float:
    """
    CLIP cosine similarity between full image embedding and text embedding.
    """

    with torch.no_grad():
        image_input = preprocess(Image.fromarray(img_rgb)).unsqueeze(0).to(DEVICE)
        img_emb = clip_model.encode_image(image_input)[0]
        img_emb = img_emb / img_emb.norm()

        tokens = clip.tokenize([text]).to(DEVICE)
        text_emb = clip_model.encode_text(tokens)[0]
        text_emb = text_emb / text_emb.norm()

        score = float((img_emb @ text_emb).item())

    return float(np.clip(score, 0.0, 1.0))


# ======================================================
# STATION 3: DINO + CLIP (PAIRWISE VERSION)
# ======================================================

def station_dino_with_clip(
    img_rgb,
    dino_model,
    dino_transform,
    normal_patch_bank,
    clip_model,
    clip_preprocess,
    normal_labels,
    anomaly_labels
):
    """
    Station 3:
    - DINO: patch-based anomaly score
    - CLIP: semantic matching against ALL labels

    Returns:
        score_dino
        score_clip_normal   (max over NORMAL_LABELS)
        score_clip_anomaly  (max over ANOMALY_LABELS)
    """

    # ---------- DINO ----------
    score_dino = station_dino_patches(
        img_rgb, dino_model, dino_transform, normal_patch_bank
    )

    # ---------- CLIP ----------
    with torch.no_grad():
        image_input = clip_preprocess(Image.fromarray(img_rgb)).unsqueeze(0).to(DEVICE)
        img_emb = clip_model.encode_image(image_input)[0]
        img_emb = img_emb / img_emb.norm()

    score_clip_normal = 0.0
    score_clip_anomaly = 0.0

    # NORMAL labels
    for _, text in normal_labels:
        tokens = clip.tokenize([text]).to(DEVICE)
        text_emb = clip_model.encode_text(tokens)[0]
        text_emb = text_emb / text_emb.norm()
        sim = float((img_emb @ text_emb).item())
        score_clip_normal = max(score_clip_normal, sim)

    # ANOMALY labels
    for _, text in anomaly_labels:
        tokens = clip.tokenize([text]).to(DEVICE)
        text_emb = clip_model.encode_text(tokens)[0]
        text_emb = text_emb / text_emb.norm()
        sim = float((img_emb @ text_emb).item())
        score_clip_anomaly = max(score_clip_anomaly, sim)

    return (
        float(np.clip(score_dino, 0, 1)),
        float(np.clip(score_clip_normal, 0, 1)),
        float(np.clip(score_clip_anomaly, 0, 1))
    )

# ======================================================
# FINAL SCORE (NOT USED YET)
# ======================================================

def compute_final_score(score_dino, score_clip_normal, score_clip_anomaly):
    """
    Final anomaly score (used only after fusion stage).
    """

    return float(np.clip(
        max(score_clip_anomaly, 1.0 - score_clip_normal, score_dino),
        0.0, 1.0
    ))


# ======================================================
# PURE CLIP SEMANTIC MATCHING (ALL LABELS)
# ======================================================

def clip_semantic_matching_all_labels(
    img_rgb: np.ndarray,
    clip_model,
    clip_preprocess,
    normal_labels,
    anomaly_labels
):
    """
    Pure CLIP semantic matching over ALL labels.
    No anomaly decision.
    """

    with torch.no_grad():
        image_input = clip_preprocess(Image.fromarray(img_rgb)).unsqueeze(0).to(DEVICE)
        image_emb = clip_model.encode_image(image_input)[0]
        image_emb = image_emb / image_emb.norm()
        image_emb = image_emb.cpu().numpy()

    scores = {}
    all_labels = normal_labels + anomaly_labels

    with torch.no_grad():
        for label_id, label_text in all_labels:
            tokens = clip.tokenize([label_text]).to(DEVICE)
            text_emb = clip_model.encode_text(tokens)[0]
            text_emb = text_emb / text_emb.norm()
            text_emb = text_emb.cpu().numpy()

            scores[label_id] = float(np.dot(image_emb, text_emb))

    best_label = max(scores, key=scores.get)
    best_score = scores[best_label]

    return scores, best_label, best_score
