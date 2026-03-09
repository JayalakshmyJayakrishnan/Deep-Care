import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

# Ensure project root is in path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.models.architecture import DeepCareMTL
from backend.models.dataset import GalarDataset

def train_model(epochs=10, batch_size=32, lr=0.001, dry_run=False):
    print("--- Initializing DeepCare MTL Training ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Setup Data Paths
    frames_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'datasets', 'Galar_Frames_1_to_10 (1)')
    labels_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'datasets', 'Galar_labels_and_metadata-20260303T071131Z-1-001', 'Galar_labels_and_metadata')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # We will use the same dataset for train/val here as a demonstration 
    # of the data loading pipeline working successfully end-to-end.
    train_dataset = GalarDataset(frames_dir=frames_dir, labels_dir=labels_dir, transform=transform)
    val_dataset = GalarDataset(frames_dir=frames_dir, labels_dir=labels_dir, transform=transform) # No separate split logic yet
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    
    # 2. Setup Model & Loss
    model = DeepCareMTL(num_anatomy_classes=3, num_pathology_classes=12)
    model = model.to(device)
    
    criterion_anatomy = nn.CrossEntropyLoss()
    # Pathology is Multi-Hot Encoded, so we MUST use BCEWithLogitsLoss
    criterion_pathology = nn.BCEWithLogitsLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    weights_dir = os.path.join(os.path.dirname(__file__), '..', 'weights')
    os.makedirs(weights_dir, exist_ok=True)
    best_model_path = os.path.join(weights_dir, 'best_model.pth')
    
    best_loss = float('inf')
    
    if dry_run:
        epochs = 1
        print("Dry run enabled. Running for 1 epoch only.")
        
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 20)
        
        # Training Phase
        model.train()
        running_loss = 0.0
        
        start_time = time.time()
        for i, (inputs, target_anatomy, target_pathology) in enumerate(train_loader):
            inputs = inputs.to(device)
            target_anatomy = target_anatomy.to(device)
            target_pathology = target_pathology.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            out_anatomy, out_pathology = model(inputs)
            
            loss_anatomy = criterion_anatomy(out_anatomy, target_anatomy)
            loss_pathology = criterion_pathology(out_pathology, target_pathology)
            
            loss = loss_anatomy + loss_pathology
            
            # Backward pass & optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if dry_run and i >= 2: # Very short iteration for dry run
                break
                
        train_loss = running_loss / max(1, len(train_loader))
        print(f"Train Loss: {train_loss:.4f} | Time: {time.time() - start_time:.2f}s")
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        correct_anatomy = 0
        correct_pathology = 0
        total = 0
        
        with torch.no_grad():
            for i, (inputs, target_anatomy, target_pathology) in enumerate(val_loader):
                inputs = inputs.to(device)
                target_anatomy = target_anatomy.to(device)
                target_pathology = target_pathology.to(device)
                
                out_anatomy, out_pathology = model(inputs)
                
                loss_anatomy = criterion_anatomy(out_anatomy, target_anatomy)
                loss_pathology = criterion_pathology(out_pathology, target_pathology)
                
                loss = loss_anatomy + loss_pathology
                val_loss += loss.item()
                
                # Anatomy Accuracy
                _, pred_anatomy = torch.max(out_anatomy, 1)
                correct_anatomy += (pred_anatomy == target_anatomy).sum().item()
                
                # Pathology Accuracy (Multi-label BCE thresholding)
                pred_pathology = (torch.sigmoid(out_pathology) > 0.5).float()
                # Consider it "correct" if the multi-hot prediction matches the multi-hot target exactly across all 12 classes
                correct_pathology += (pred_pathology == target_pathology).all(dim=1).sum().item()
                
                total += target_anatomy.size(0)
                
                if dry_run and i >= 2:
                    break
                    
        val_loss = val_loss / max(1, len(val_loader))
        acc_anatomy = 100 * correct_anatomy / max(1, total)
        acc_pathology = 100 * correct_pathology / max(1, total)
        
        print(f"Val Loss: {val_loss:.4f} | Anatomy Acc: {acc_anatomy:.2f}% | Pathology Exact Match Acc: {acc_pathology:.2f}%")
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            print(f"--> Val Loss improved! Saving model to {best_model_path}")
            torch.save(model.state_dict(), best_model_path)
            
    print("\n--- Training Complete ---")
    print(f"Best Validation Loss: {best_loss:.4f}")
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train DeepCare MTL Model')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--dry-run', action='store_true', help='Run a quick dry run')
    args = parser.parse_args()
    
    train_model(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, dry_run=args.dry_run)
