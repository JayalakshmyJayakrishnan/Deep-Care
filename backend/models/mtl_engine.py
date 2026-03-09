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
        
        # Load weights if available
        weights_path = os.path.join(os.path.dirname(__file__), '..', 'weights', 'best_model.pth')
        if os.path.exists(weights_path):
            try:
                self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
                print(f"MTLEngine: Loaded trained weights from {weights_path}")
            except Exception as e:
                print(f"MTLEngine: Failed to load weights ({e}). Using uninitialized model.")
        else:
            print("MTLEngine: No trained weights found. Using uninitialized model.")
            
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
