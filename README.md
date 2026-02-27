# 🚦 UrbanRiskGPT

## Training-Free Risk-Relevant Anomaly Detection in Urban Road Scenes

---

## Abstract

Urban safety monitoring requires robust detection of rare, unexpected, and context-dependent risk events in road environments. Traditional supervised approaches rely on large labeled datasets of accident scenarios, which are costly to collect and inherently incomplete.

This repository presents two complementary training-free vision-language systems for detecting risk-relevant anomalies in urban road scenes:

- Zero-Shot Vision-LLM Reasoning System  
- Few-Shot Modular Vision-Semantic Deviation Pipeline  

Both systems operate entirely at inference time without task-specific fine-tuning.  
The work explores whether structured semantic reasoning and self-supervised visual deviation modeling can enable reliable anomaly detection without supervised training.

---

## Contributions

- Training-free risk detection framework for urban road scenes  
- Structured Vision-LLM reasoning pipeline producing:
  - Scene understanding  
  - Object-level semantic analysis  
  - Risk scoring (1–10 severity scale)  
  - Spatial localization (bounding boxes + polygons)  
  - Natural-language explanation  
- Modular DINO–CLIP–GPT pipeline for visual deviation–based anomaly detection  
- Unified evaluation protocol across inference-only configurations  
- Comparison between reasoning-based and deviation-based paradigms  

---

## System Overview

### 1️⃣ Zero-Shot Vision-LLM System

A structured GPT-based reasoning engine processes each image using a controlled prompt format.

For each scene, the system produces:

- Scene type (road / sidewalk / urban)  
- Scene summary  
- Detected objects  
- Anomaly identification  
- Risk targets  
- Risk severity score (1–10)  
- Bounding boxes  
- Polygon segmentation  
- Explicit reasoning  

The system performs fully structured semantic interpretation without training.

---

### 2️⃣ Few-Shot Modular Pipeline

A deviation-driven anomaly detector composed of:

- DINO – Patch-level structural deviation detection  
- CLIP – Semantic anomaly validation  
- GPT – Risk explanation generation  

Final anomaly score combines:

- DINO deviation score  
- CLIP semantic confidence  

---

## Repository Structure


UrbanRiskGPT/
│
├── src/
│ ├── gpt_only/
│ │ ├── run_gpt_batch.py
│ │ ├── evaluate_zero_shot.py
│ │ └── score_reasoning.py
│
UrbanZS/
│ ├── compute_statistics.py
│ ├── anomaly_pipeline.py
│ ├── evaluate_model.py
│ ├── analysis_arch3_dino_clip_semantic.py



---

### Zero-Shot Pipeline

#### Step 1 — Prepare Manifest File

Each line in `manifest.jsonl`:

```json
{
  "sample_id": "img_001",
  "frame_path": "images/img_001.jpg",
  "label": 1
}

python run_gpt_batch.py \
  --manifest path/to/manifest.jsonl \
  --output path/to/results.jsonl \
  --model gpt-5.1
