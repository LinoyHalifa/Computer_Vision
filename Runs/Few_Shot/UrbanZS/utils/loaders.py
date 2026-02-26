import os
from pathlib import Path

import torch
import clip
from openai import OpenAI

try:
    import timm
except ImportError:
    timm = None

try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
except ImportError:
    sam_model_registry = None
    SamAutomaticMaskGenerator = None

from utils.common import DEVICE

# Base directory of the project (UrbanZH)
BASE_DIR = Path(__file__).resolve().parent.parent

# Model paths (relative)
YOLO_WEIGHTS = BASE_DIR / "models" / "yolov10b.pt"
SAM_MODEL_TYPE = "vit_h"
SAM_CHECKPOINT = BASE_DIR / "models" / "sam_vit_h_4b8939.pth"


def load_yolo():
    """Load YOLO model for object detection."""
    from ultralytics import YOLO
    if not YOLO_WEIGHTS.is_file():
        raise FileNotFoundError(f"YOLO weights not found at: {YOLO_WEIGHTS}")
    model = YOLO(str(YOLO_WEIGHTS))
    return model


def load_clip():
    """Load CLIP model + preprocess."""
    model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    return model, preprocess


def precompute_text_embeddings(clip_model, labels):
    """
    Compute CLIP embeddings for any list of labels:
    labels = [(id, description), ...]
    """
    import torch
    import clip

    texts = [desc for (_id, desc) in labels]
    tokens = clip.tokenize(texts).to("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        text_features = clip_model.encode_text(tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    return text_features.cpu().numpy()



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
    if sam_model_registry is None or SamAutomaticMaskGenerator is None:
        raise ImportError(
            "segment_anything is not installed. Install it and its dependencies."
        )

    if not SAM_CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"SAM checkpoint not found at {SAM_CHECKPOINT}. "
            f"Download it and place it under models/."
        )

    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=str(SAM_CHECKPOINT))
    sam.to(device=DEVICE)
    mask_generator = SamAutomaticMaskGenerator(sam)
    return mask_generator


def build_openai_client():
    """Create OpenAI client (OPENAI_API_KEY must be set)."""
    return OpenAI()
