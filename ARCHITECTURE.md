# Deep Care System Architecture

## Overview
Deep Care employs a sequential processing pipeline designed for high-throughput video analysis of WCE data.

## Pipeline Components

### 1. Ingestion & Preprocessing
*   **Input:** Raw Video / Image Sequence
*   **Module:** `backend.utils.preprocessing`
*   **Operations:** Frame extraction (5fps), Resizing to 224x224, CLAHE enhancement.

### 2. Quality Gate (Phase 1)
*   **Module:** `backend.models.quality_gate`
*   **Logic:** Binary classification (Informative vs. Non-Informative).
*   **Current Impl:** Laplacian Variance heuristic (Placeholder for MobileNetV4).

### 3. Multi-Task Learning Engine (Phase 2)
*   **Module:** `backend.models.mtl_engine`
*   **Tasks:**
    *   **Anatomy:** Stomach, Small Bowel, Colon.
    *   **Pathology:** 14-class detection (Bleeding, Polyps, Ulcers, etc.).
*   **Backbone:** EfficientNet-Edge (Mocked).

### 4. Temporal Refinement (Phase 3)
*   **Module:** `backend.models.temporal`
*   **Logic:** Sliding Window Smoothing / Viterbi Algorithm.
*   **Purpose:** Remove flickering predictions (e.g., Normal -> Polyp -> Normal).

### 5. Explainability (Phase 4)
*   **Module:** `backend.models.xai`
*   **Output:** Grad-CAM++ Heatmaps overlay.
*   **Purpose:** Visual verification of model focus.

## Data Flow
`Video` -> `Preprocessor` -> `Frames` -> `QualityGate` -> `MTLEngine` -> `TemporalRefiner` -> `XAIGenerator` -> `JSON Report + Heatmaps`

## API Endpoints
*   `POST /api/upload`: Accepts file, processes via pipeline, returns comprehensive JSON report.
