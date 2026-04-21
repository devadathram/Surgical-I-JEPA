# Surgical I-JEPA

This repository contains the implementation of a self-supervised surgical foundation model. The project shifts from traditional pixel-level reconstruction to **semantic representation learning** using the Image-based Joint-Embedding Predictive Architecture (I-JEPA), tailored for laparoscopic surgical video analysis.

## 📌 Project Overview

Traditional surgical AI models often struggle with the dynamic and "noisy" environment of an operating room (smoke, reflections, fluids). This project addresses these challenges by:
- **Prioritizing Semantics:** Moving beyond pixel reconstruction (like MAE) to predict high-level latent representations.
- **Surgical Latent Space:** Creating a manifold where the AI learns the "Surgical Grammar"—the logic of anatomy and tool interactions.
- **Zero-Shot Potential:** Utilizing foundational models like U-Net, SAM, and YOLO within a constrained learning pipeline.

## 🚀 Key Features

### 1. Advanced Representation Learning (I-JEPA)
Instead of predicting pixels, our I-JEPA implementation predicts the latent representation of target blocks from context blocks. This ensures the model captures the **context and intent** of surgical phases.
- **Mean Testing Similarity:** 87.4%
- **Mean Training Similarity:** 94.2%

### 2. Constrained Learning Pipeline
A specialized pipeline developed to handle the unique constraints of surgical data:
- **Multi-scale Masking:** Prevents the model from relying on local textures.
- **Temporal Stability:** Optimized to capture invariant features across high-redundancy video frames.

### 3. Workflow & Anomaly Foundation
The resulting embeddings serve as a robust visual bottleneck, laying the groundwork for integrating dynamic temporal context.

## 📊 Results Summary

The efficacy of the model is quantified through **Cosine Similarity** in the latent space, measuring the mathematical alignment between the model's internal predictions and the structural reality of the surgery.

| Metric | Value |
| :--- | :--- |
| Training Cosine Similarity | 94.2% |
| Testing Cosine Similarity | 87.4% |

## 🛠️ Tech Stack

- **Backbone:** Vision Transformer (ViT)
- **Architecture:** I-JEPA (Image Joint-Embedding Predictive Architecture)
- **Tools:** PyTorch, OpenCV, Scikit-learn
- **Dataset:** Cholec80 / Cholec50


---

*This project was developed as part of a research internship at the Instituto Superior Técnico (IST-ID), Lisbon.*
