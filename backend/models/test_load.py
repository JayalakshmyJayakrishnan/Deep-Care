import torch
import os

models_dir = r"d:\MINOR\pillcam_platform\backend\models"
files = [
    "gi_full_model.pt (1).zip",
    "gi_model.pth (1).zip",
    "gi_model_scripted.pt (1).zip"
]

for f in files:
    path = os.path.join(models_dir, f)
    print(f"--- Loading {f} ---")
    try:
        # Try jit load
         data = torch.jit.load(path, map_location='cpu')
         print("Successfully loaded as TorchScript!")
    except Exception as e1:
         try:
             data = torch.load(path, map_location='cpu', weights_only=False)
             if isinstance(data, dict):
                 print("Successfully loaded as state_dict (dict)!")
                 print("Keys:", list(data.keys())[:5])
             else:
                 print(f"Successfully loaded via torch.load. Type: {type(data)}")
         except Exception as e2:
             print(f"Failed to load: JIT error ({e1}), Load error ({e2})")

