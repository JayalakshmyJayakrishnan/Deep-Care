import os
import torch

import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from backend.models.architecture import DeepCareMTL

class MTLEngine:
    def __init__(self):
        self.device = torch.device("cpu") # Use CPU for broad compatibility here
        
        # Load the Real Model Architecture
        self.model = DeepCareMTL(num_anatomy_classes=3, num_pathology_classes=12)
        
        # Load the official pre-trained weights
        weights_path = os.path.join(os.path.dirname(__file__), 'gi_model.pth')
        if not os.path.exists(weights_path):
            raise RuntimeError(f"Critical Failure: The pathology model weights were not found at '{weights_path}'. Ensure the required files are present.")
            
        try:
            # We enforce weights_only=False because standard PyTorch save/load via zip needs it, and we control the files locally
            state_dict = torch.load(weights_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(state_dict, strict=False)
            print(f"MTLEngine: successfully instantiated and loaded trained weights from {weights_path}")
        except Exception as e:
            raise RuntimeError(f"Critical Failure: Failed to interpret or load the state_dict from weights file. Details: {e}")
            
        self.model.eval()
        self.model.to(self.device)
        
        # Define Classes
        self.anatomical_classes = ["Stomach", "Small Bowel", "Colon"]
        self.lesion_classes = [
            "Normal", "Angioectasia", "Bleeding", "Chylous Cyst", 
            "Depressed", "Diverticulum", "Erosion", "Lymphangiectasia", 
            "Polyp", "Red Spots", "Ulcer", "Stricture"
        ]
        
        # Standard ImageNet Transforms (plus resizing to be safe)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225]),
        ])

    def predict(self, frame_bgr):
        """
        Runs inference using the real EfficientNet-B0 backbone.
        """
        # Convert BGR to RGB (OpenCV -> PyTorch)
        frame_rgb = frame_bgr[..., ::-1].copy()
        
        # Transform
        input_tensor = self.transform(frame_rgb).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            anatomy_logits, pathology_logits = self.model(input_tensor)
            
            # Softmax to get probabilities
            anatomy_probs = F.softmax(anatomy_logits, dim=1)
            pathology_probs = F.softmax(pathology_logits, dim=1)
            
            # Get Predictions
            anatomy_idx = torch.argmax(anatomy_probs, dim=1).item()
            pathology_idx = torch.argmax(pathology_probs, dim=1).item()
            
            path_conf = pathology_probs[0][pathology_idx].item()
            
            anatomy_name = self.anatomical_classes[anatomy_idx % len(self.anatomical_classes)]
            pathology_name = self.lesion_classes[pathology_idx % len(self.lesion_classes)]

        return {
            "anatomical": anatomy_name,
            "pathology": pathology_name,
            "pathology_confidence": float(path_conf),
            # Pass gradients/tensor info if needed for XAI, 
            # but usually XAI needs the gradient enabled, so we handle XAI separatedly.
            "input_tensor": input_tensor 
        }

    def get_model(self):
        return self.model
