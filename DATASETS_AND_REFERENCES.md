# Recommended Datasets for PillCam/Endoscopy Project

For training advanced ML models (CNNs like ResNet, YOLO, or EfficientNet), you will need labeled medical imaging data. Here are the top open-source datasets:

## 1. Kvasir Dataset
* **Description**: One of the most popular multi-class datasets for GI diseases. Contains images of anatomical landmarks, pathological findings (esophagitis, polyps, ulcers), and endoscopic interventions.
* **Size**: ~8,000 images (8 classes).
* **Reference**: [https://datasets.simula.no/kvasir/](https://datasets.simula.no/kvasir/)

## 2. CVC-ClinicDB
* **Description**: Focused on polyp detection in colonoscopy videos. Contains frames with ground truth (masks) for segmentation tasks.
* **Reference**: [https://www.kaggle.com/datasets/balraj98/cvcclinicdb](https://www.kaggle.com/datasets/balraj98/cvcclinicdb)

## 3. KID Dataset (Wireless Capsule Endoscopy)
* **Description**: Specifically designed for Knowledge-based identification of WCE images.
* **Description**: Specifically designed for Knowledge-based identification of WCE images.
* **Reference**: [Mendeley Data - KID Dataset](https://data.mendeley.com/)

## 4. Galar Dataset (2025)
* **Description**: A large multi-label video capsule endoscopy dataset. Contains high-quality video sequences suitable for training temporal models.
* **Reference**: [Figshare Link](https://plus.figshare.com/articles/dataset/Galar_-_a_large_multi-label_video_capsule_endoscopy_dataset/25304616?file=44724651)
* **Integration**: Run `python backend/scripts/download_galar.py` to download and unzip into `datasets/galar`.

## How to integrate into this project
1. **Download**: Download the Kvasir dataset.
2. **Train**: Train a PyTorch/TensorFlow model to classify images as "Normal", "Bleeding", "Ulcer", "Polyp".
3. **Export**: Save the model weights (e.g., `model.pth`).
4. **Update Backend**: Modify `backend/app.py` to load the model and use it for prediction instead of the current color-based heuristic.
