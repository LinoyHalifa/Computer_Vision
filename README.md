Urban Risk Detection – Training-Free Vision Systems

This repository presents two complementary training-free approaches for detecting risk-relevant anomalies in urban environments using vision-language reasoning.

The project compares:

Zero-Shot Vision-LLM system

Few-Shot modular vision-language pipeline

Both systems operate at inference time without task-specific training.

Overview

Urban safety monitoring often relies on supervised models trained on labeled accident data. However, many real-world risks are rare, unexpected, or context-dependent.

This project explores training-free perception using two paradigms:

Zero-Shot System

A Vision-LLM (GPT-based) that directly interprets road scenes and produces:

Scene understanding

Object detection (semantic)

Anomaly identification

Risk scoring

Spatial localization (boxes + polygons)

Natural-language reasoning

Few-Shot System

A modular pipeline combining:

DINO → visual deviation detection

CLIP → semantic alignment

GPT → post-hoc explanation

Repository Structure
Zero-Shot Pipeline

Location:
UrbanRiskGPT/src/gpt_only/

Main scripts:

run_gpt_batch.py
Runs GPT analysis on a dataset (multiple images)

evaluation
Location:
UrbanRiskGPT/src/gpt_only/

evaluate_zero_shot.py
Computes evaluation metrics (Accuracy, F1, ROC-AUC, PR-AUC)

score_reasoning.py
computes the scores of the reasoning 


Few-Shot Pipeline (Station 3 Based)

Location:
UrbanZS/

Main scripts:

compute_statistics.py
Builds statistical calibration for DINO scores using normal images

anomaly_pipeline.py
Runs the system on a single image (demo mode)

evaluation

evaluate_model.py
Runs the system on full dataset and computes metrics

analysis_arch3_dino_clip_semantic.py
arch3_dino_clip_semantic_analysis.py
computes the metrices of the semantic part


Zero-Shot System – How It Works

The system sends each image to GPT-5.1 with a structured prompt.

For each image, the model returns:

Scene type (road / sidewalk)

Scene summary

Relevant objects

Detected anomalies

Risk score (1-10)

Risk targets

Bounding boxes

Polygon segmentation

Reasoning

This creates a fully structured semantic interpretation without training.

Zero-Shot – How to Run

Step 1 — Prepare a manifest file:

Each line in manifest.jsonl:

{
"sample_id": "img_001",
"frame_path": "images/img_001.jpg",
"label": 1
}

Step 2 — Run inference:

python run_gpt_batch.py
--manifest path/to/manifest.jsonl
--output path/to/results.jsonl
--model gpt-5.1

Step 3 — Evaluate:

python metrics_zero_shot.py

This produces:

Accuracy

Precision / Recall / F1

ROC curve

PR curve

Confusion matrix

Few-Shot System – How It Works

This pipeline detects anomalies using visual deviation rather than language reasoning alone.

Station 3 Components

DINO:
Detects structural irregularities via patch similarity

CLIP:
Matches image to semantic labels

GPT:
Explains detected anomaly

The final anomaly score is computed from:

DINO deviation

CLIP semantic confidence

Few-Shot – How to Run
Step 1 — Build DINO statistics (important)

Place normal images in:

normal_images/

Then run:

python compute_stats.py

This generates:
statistics.json

Used to normalize anomaly scores.

Step 2 — Run on a single image (demo)

Place image in:

input_images/

Then run:

python analyze_image_station3.py

This produces:

JSON result

On-screen visualization

GPT explanation

Step 3 — Full evaluation

Prepare:

evaluation/
├── normal/
├── anomaly/

Then run:

python evaluation.py

This produces:

metrics.json

roc_curve.png

pr_curve.png

confusion_matrix.png

results.csv

If you use this repository, please cite:

Training-Free Recognition of Risk-Relevant Situations in Urban Road Scenes Using Vision-Language Models
Halifa et al.

Author

Linoy Halifa
Ezra Ella
M.Sc. Intelligent Systems
