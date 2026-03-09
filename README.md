# Deep Care: Intelligent Wireless Capsule Endoscopy
> Formerly PillCam Platform

Deep Care is a Transparent Clinical Decision Support Tool designed to revolutionize Wireless Capsule Endoscopy (WCE) diagnostics. It transitions from a "black-box" AI to an explainable, multi-task learning system.

## Project Phases & Architecture

### Phase 1: Data Pre-processing
- **Objective:** Normalize lighting and texture.
- **Tech:** OpenCV, CLAHE (Contrast Limited Adaptive Histogram Equalization).
- **Status:** Implemented (`backend/utils/preprocessing.py`).

### Phase 2: Automated Quality Validation (The Gatekeeper)
- **Objective:** Filter out non-diagnostic frames (bubbles, debris).
- **Tech:** MobileNetV4 (Placeholder logic implemented).
- **Status:** Implemented (`backend/models/quality_gate.py`).

### Phase 3: Multi-Task Learning (MTL) Engine
- **Objective:** Simultaneous Anatomical Localization and Pathology Classification.
- **Tech:** EfficientNet-Edge Backbone.
- **Status:** Architecture Implemented (`backend/models/mtl_engine.py`).

### Phase 4: Temporal Consistency
- **Objective:** Enforce logical consistency across video frames using Sliding Window/Viterbi.
- **Status:** Implemented (`backend/models/temporal.py`).

### Phase 5: Explainable AI (XAI)
- **Objective:** Generate Saliency Heatmaps (Grad-CAM++) for clinician trust.
- **Status:** Implemented (`backend/models/xai.py`).

## Tech Stack
- **Backend:** Python, Flask, OpenCV, NumPy
- **Frontend:** React, Vite
- **AI/ML:** Modular pipeline architecture designed for MobileNetV4 & EfficientNet.

## Getting Started

1. **Start the Backend:**
   ```bash
   cd backend
   python app.py
   ```

2. **Start the Frontend:**
   ```bash
   cd frontend
   npm run dev
   ```

3. **Usage:**
   - Upload a video or image sequence.
   - The system processes it through the 5-step pipeline.
   - View the "Deep Care Report" with abnormalities and heatmaps.
