import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from backend.models.architecture import DeepCareMTL

path_pth = r"d:\MINOR\pillcam_platform\backend\models\gi_model.pth"
path_full = r"d:\MINOR\pillcam_platform\backend\models\gi_full_model.pt"
path_scripted = r"d:\MINOR\pillcam_platform\backend\models\gi_model_scripted.pt"

dummy = torch.randn(1, 3, 224, 224)

print("\n--- Testing gi_model.pth with DeepCareMTL ---")
mtl = DeepCareMTL()
try:
    sd = torch.load(path_pth, map_location='cpu', weights_only=False)
    # try strict=False
    missing, unexpected = mtl.load_state_dict(sd, strict=False)
    print("Loaded with strict=False!")
    print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    out = mtl(dummy)
    print(f"Forward pass successful. Outputs: {len(out)} tensors.")
except Exception as e:
    print(f"Error: {e}")

print("\n--- Testing gi_model_scripted.pt ---")
try:
    sc = torch.jit.load(path_scripted, map_location='cpu')
    out2 = sc(dummy)
    if isinstance(out2, tuple):
        print(f"Scripted returned tuple of len {len(out2)}")
    elif isinstance(out2, torch.Tensor):
        print(f"Scripted returned tensor of shape {out2.shape}")
    else:
        print(f"Scripted returned {type(out2)}")
except Exception as e:
    print(f"Error: {e}")

print("\n--- Testing gi_full_model.pt ---")
try:
    full = torch.load(path_full, map_location='cpu', weights_only=False)
    if isinstance(full, torch.nn.Module):
        out3 = full(dummy)
        if hasattr(out3, '__len__'):
            print(f"Full model returned len {len(out3)}")
        else:
            print(f"Full model returned tensor shape {out3.shape}")
    else:
        print(f"Type of full model diff: {type(full)}")
except Exception as e:
    print(f"Error: {e}")
