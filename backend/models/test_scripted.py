import torch
import cv2
import numpy as np

scripted_path = r"d:\MINOR\pillcam_platform\backend\models\gi_model_scripted.pt"
print(f"Loading {scripted_path}...")
model = torch.jit.load(scripted_path, map_location='cpu')
model.eval()

# Dummy input
dummy = torch.randn(1, 3, 224, 224)
try:
    anatomy_logits, pathology_logits = model(dummy)
    print("Scripted model inference successful!")
    print(f"Anatomy shape: {anatomy_logits.shape}, Pathology shape: {pathology_logits.shape}")
except Exception as e:
    print(f"Error during forward pass: {e}")
