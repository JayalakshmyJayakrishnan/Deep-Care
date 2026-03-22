# Deep-Care
we deeply care about your gut ツ ♡

### A Quality-Aware Multi-Task Learning Framework for Efficient and Explainable Gastrointestinal Abnormality Detection

> **Adi Shankara Institute of Engineering and Technology**  
> Department of Data Science Engineering  
> Guided by: Ms. Sabitha M.G., Assistant Professor

---

## Overview

DeepCare is a quality-aware, context-driven multi-task learning framework for reliable and interpretable detection of gastrointestinal (GI) abnormalities from Wireless Capsule Endoscopy (WCE) video data.

WCE produces tens of thousands of frames per patient examination. Manual review is tedious, cognitively demanding, and prone to missed findings. DeepCare automates this pipeline — filtering noise, classifying pathologies, enforcing temporal consistency, and delivering explainable outputs through a clinical web dashboard.

---

## Pipeline

```
WCE Video Input
    │
    ▼
Image Preprocessing (Frame Extraction + CLAHE + Resize)
    │
    ▼
Quality Gatekeeper (MobileNetV4 + Laplacian Variance)
    │
    ▼
Multi-Task Learning Engine (EfficientNet Backbone)
    ├── Anatomical Localization (Stomach / Small Bowel / Colon)
    └── Pathological Classification (Bleeding / Ulcer / Polyp / Erosion)
    │
    ▼
Temporal Consistency Module (Sliding Window + Viterbi Smoothing)
    │
    ▼
Explainability Module (Grad-CAM++ Heatmaps)
    │
    ▼
Clinical Dashboard (React + FastAPI + PostgreSQL)
```

---

## Key Features

| Module | Description |
|--------|-------------|
| **Quality Gatekeeper** | MobileNetV4 binary classifier + Laplacian variance check to auto-filter blurry, bubble-contaminated, and debris frames |
| **Multi-Task Learning** | EfficientNet backbone with dual heads for simultaneous anatomical localization and pathology classification |
| **Temporal Consistency** | Viterbi-based smoothing over sliding-window frame groups to eliminate prediction flicker |
| **Explainable AI** | Grad-CAM++ heatmaps superimposed on WCE frames for clinician-verifiable AI reasoning |
| **Clinical Dashboard** | FastAPI + React web interface for video upload, real-time inference, heatmap review, and report export |

---

## 🗃️ Datasets

| Dataset | Modality | Frames | Annotations |
|---------|----------|--------|-------------|
| Kvasir-Capsule | WCE | 47,238 | Polyps, Bleeding, Ulcers, Erosions, Normal |
| KvasirSEG | WCE | 1,000 | Polyps, Bleeding Lesions |
| Galar Dataset | WCE | 235,000+ | Ulcers, Erosions, Bleeding |

All datasets are in JPEG format.

---

## System Architecture

The system follows a four-tier architecture:

- **Client Tier** — Web-based frontend (HTML / CSS / JS)
- **API Tier** — FastAPI backend handling CORS, routing, and request validation
- **Processing Tier** — OpenCV frame extraction, CLAHE preprocessing, MobileNetV4 quality filtering
- **AI/Inference Tier** — EfficientNet-MTL for segmentation and Grad-CAM++ heatmap generation
- **Output Tier** — Radiomics extraction and structured clinical diagnostic report generation

---

## Modules

### Module 1 — Image Preprocessing & Quality Gatekeeper
- WCE video frames extracted via OpenCV
- CLAHE applied for contrast enhancement
- MobileNetV4 binary classifier discards non-diagnostic frames
- Laplacian variance check filters severe motion blur

### Module 2 — Multi-Task Learning Engine
- **Task 1:** Anatomical localization → stomach / small bowel / colon
- **Task 2:** Pathological classification → bleeding / ulcers / polyps / erosions
- Joint learning captures spatial context and pathological features simultaneously
- Trained with Adam optimizer (lr=0.0001), batch size 16, 20+ epochs, ImageNet transfer learning

### Module 3 — Temporal Consistency (Viterbi)
- Sliding-window grouping captures short-term temporal dependencies
- Viterbi algorithm applies anatomical transition constraints across the full video sequence
- Eliminates frame-level prediction flicker and transient misclassifications

### Module 4 — Explainable AI & Clinical Dashboard
- Grad-CAM++ generates visual heatmaps highlighting diagnostically influential regions
- Heatmaps overlaid on original WCE frames for clinical interpretability
- Structured diagnostic output: organ label, anomaly flag, radiomics statistics
- Results rendered on the React dashboard with per-class confidence scores

---

## Technology Stack

| Layer | Technologies |
|-------|-------------|
| Frontend | HTML, CSS, JavaScript, React |
| Backend | Python 3, FastAPI, Uvicorn |
| AI / ML | PyTorch (EfficientNet, MobileNetV4), OpenCV |
| Database | PostgreSQL |
| Data Formats | JPEG, NIfTI (.nii.gz) |

---

## 📊 Results

### Performance vs. Existing Methods

| Method | Modality | Accuracy (%) | F1 Score (%) |
|--------|----------|-------------|-------------|
| Abd El-Ghany et al. (CNN) | WCE | 88.2 | 79.4 |
| Chen et al. (CNN-RNN) | WCE | 87.6 | 80.1 |
| Bingol & Yildirim | WCE | 86.3 | 78.9 |
| ResNet-based Segmentation | WCE/MRI | 85.9 | 77.2 |
| SegResNet (MONAI) | WCE | 87.5 | 80.8 |
| **DeepCare (Proposed)** | **WCE + MRI** | **90.4** | **83.6** |

### Key Metrics
- **Accuracy:** >90% on Kvasir-Capsule dataset
- **Weighted F1 Score:** 83.6% across all 10 GI abnormality classes
- **Inference Latency:** <33ms per frame
- **Lesion Types Supported:** 14

---

## 🔭 Future Scope

- Train on larger, more diverse WCE datasets covering a wider range of GI pathologies
- Real-time on-device inference for next-generation capsule endoscopy hardware
- Cloud-based and GPU-accelerated deployment with REST API integration into hospital PACS systems
- Federated learning for privacy-preserving multi-hospital model training

---

## Team

| Name | Role |
|------|------|
| Jayalakshmy Jayakrishnan | Developer |
| Akhiyar Muhammed | Developer |
| Sreevidya V | Developer |
| Devika Viswam | Developer |

**Guide:** Ms. Sabitha M.G., Assistant Professor, Department of Data Science Engineering  
**Institution:** Adi Shankara Institute of Engineering and Technology (Approved by AICTE, Affiliated to APJ Abdul Kalam Technological University)

---

## References

1. S. Abd El-Ghany et al., "An Accurate Deep Learning-Based CAD System for GI Disease Detection Using WCE Image Analysis," *IEEE Access*, vol. 12, 2024.
2. J. Chen et al., "Automated Capsule Endoscopy Recognition Based on CNNs," *Medical Image Analysis*, vol. 97, 2024.
3. M. R. Islam, M. Qaraqe, and E. Serpedin, "Automatic Detection of GI Diseases Using WCE Images," *IEEE J. Biomed. Health Inform.*, 2022.
4. Jain et al., "Hybrid CNN with Meta Feature Learning for Abnormality Detection in WCE Images," *Diagnostics*, vol. 12, no. 5, 2022.
5. M. J. Cardoso et al., "MONAI: An open-source framework for deep learning in healthcare," *IEEE J. Biomed. Health Inform.*, 2022.

---

*Presented on 06-03-2026 | Department of Data Science Engineering*
