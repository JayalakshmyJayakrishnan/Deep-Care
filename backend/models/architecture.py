
import torch
import torch.nn as nn
from torchvision import models

class DeepCareMTL(nn.Module):
    def __init__(self, num_anatomy_classes=3, num_pathology_classes=12):
        super(DeepCareMTL, self).__init__()
        
        # 1. Backbone: EfficientNet-B0 (Pre-trained on ImageNet)
        # We use this as a feature extractor.
        weights = models.EfficientNet_B0_Weights.DEFAULT
        self.backbone = models.efficientnet_b0(weights=weights)
        
        # Remove the original classifier head
        # The classifier in EfficientNet is a Sequential block. 
        # We want the output of the dropout layer before the final Linear.
        # EfficientNet structure: features -> avgpool -> classifier
        # We'll tap into 'avgpool'.
        
        # Feature dimension for B0 is 1280
        self.feature_dim = 1280
        
        # 2. Shared Embeddings
        # We use the backbone's feature extraction capabilities
        
        # 3. Task-Specific Heads based on "Hard Parameter Sharing"
        
        # Task A: Anatomical Localization Head
        self.anatomy_head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_anatomy_classes)
        )
        
        # Task B: Pathology Classification Head
        self.pathology_head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_pathology_classes)
        )
        
    def forward(self, x):
        # Extract features
        # x shape: [Batch, 3, 224, 224]
        features = self.backbone.features(x)         # [Batch, 1280, 7, 7]
        features = self.backbone.avgpool(features)   # [Batch, 1280, 1, 1]
        features = torch.flatten(features, 1)        # [Batch, 1280]
        
        # Multi-Task Inference
        anatomy_logits = self.anatomy_head(features)
        pathology_logits = self.pathology_head(features)
        
        return anatomy_logits, pathology_logits

    def getLastConvLayer(self):
        # Used for Grad-CAM
        # Returns the last convolutional layer of the features block
        return self.backbone.features[-1]
