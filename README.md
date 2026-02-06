# IMME
Interactive Multimodal Embedding Explorer with Jeppa

<div align="center">

# Multimodal Embedding Explorer (MMEE)

### _Visualizing • Aligning • Evaluating Vision–Language Embeddings_

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Flask](https://img.shields.io/badge/Backend-Flask-black)
![Next.js](https://img.shields.io/badge/Frontend-Next.js-black)
![GSoC](https://img.shields.io/badge/GSoC-Ready-orange)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

---

## Overview

**Multimodal Embedding Explorer (MMEE)** is a research-focused framework for **analyzing, visualizing, and evaluating vision–language embeddings**.

It enables **side-by-side comparison** between:
- **CLIP** — contrastive learning
- **VL-JEPA-inspired** — predictive learning

MMEE emphasizes **interpretability, alignment quality, anomaly detection, and evaluation**, rather than training massive models from scratch.

---

## ✨ What Makes MMEE Special?

- Deep **embedding interpretability**
- Research-grade **evaluation metrics**
- Contrastive vs Predictive **learning comparison**
- Interactive **2D joint visualizations**
- GSoC-ready **modular architecture**

---

## Architecture Overview
<img width="671" height="673" alt="image" src="https://github.com/user-attachments/assets/583dccfa-c874-43a9-ad7d-10380acbebdd" />
<img width="861" height="724" alt="image" src="https://github.com/user-attachments/assets/aa1ea048-9057-4024-8504-c1eb6609bf68" />
<img width="693" height="111" alt="image" src="https://github.com/user-attachments/assets/18a284a1-1499-4514-8e93-abeef09f7797" />

---

## Conceptual Pipeline (Animated Flow)

Image + Text Dataset
↓
Embedding Models
├─ CLIP (Contrastive)
└─ VL-JEPA-inspired (Predictive)
↓
High-D Embeddings (512D / 768D)
↓
Quantitative Analysis
├─ Cosine Similarity
├─ Procrustes Alignment
├─ Outlier Detection
└─ Fusion Scorer
↓
PCA / t-SNE / UMAP
(Visualization Only)
↓
Interactive Dashboard


---

## Embedding Models

<details>
<summary><b>CLIP — Contrastive Learning</b></summary>

- Learns by **pulling matching image–text pairs together**
- Pushes mismatched pairs apart using negatives
- Strong baseline for vision–language alignment

</details>

<details>
<summary><b>VL-JEPA-Inspired — Predictive Learning</b></summary>

- Learns by **predicting missing semantic representations**
- No explicit negative samples
- Captures deeper contextual understanding

> Implemented as a **JEPA-inspired encoder for inference & analysis**, not full-scale training.

</details>

---

## Dimensionality Reduction (Visualization Only)

> These methods **do NOT change embeddings** — they only help humans see them.

| Method | Purpose |
|------|--------|
| **PCA** | Global variance structure |
| **t-SNE** | Local neighborhood clarity |
| **UMAP** | Best balance (default) |

---

## Alignment & Similarity Analysis

- **Cosine similarity** in original embedding space  
- **Orthogonal Procrustes alignment**
- Residual error as alignment quality signal

---

## Outlier Detection (Per-Class)

Used to identify **noisy, misaligned, or anomalous samples**.

- Isolation Forest  
- Local Outlier Factor (LOF)  
- kNN distance quantiles  
- DBSCAN  

Outputs:
- Anomaly scores  
- Binary outlier flags  

---

## Fusion Scorer (Decision Layer)

A **Logistic Regression model with cross-validation** combines multiple weak signals into a strong prediction.

**Input Signals**
- Cosine similarity
- Alignment residual
- Outlier scores
- Caption length
- Embedding norms

**Output**
- Match probability
- Final decision

---

## Evaluation Metrics

- ROC Curve & AUC
- Precision–Recall Curve
- Confusion Matrix
- Ranked prediction tables

All metrics computed using **scikit-learn**.

---

## Interactive Dashboard

Built with **Next.js + Plotly.js**, featuring:

- Joint image–text 2D plots
- Connection lines showing alignment
- Hover thumbnails & captions
- Interactive ROC / PR curves
- Confusion matrices

---

##  Tech Stack

###  Backend
- Flask
- PyTorch
- NumPy, SciPy
- scikit-learn
- UMAP-learn
- pandas

### Frontend
- Next.js (React)
- Plotly.js
- Base64 image rendering

---

## Future Work

- Audio & 🎥 Video modalities
- FAISS vector search
- Bias & robustness analysis
- Additional embedding backbones



## License
MIT License
<div align="center">
Built for interpretability, research, and GSoC
</div>

