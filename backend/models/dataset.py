import os
import glob
import pandas as pd
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class GalarDataset(Dataset):
    """
    PyTorch Dataset for the authentic Galar Video Capsule Endoscopy dataset.
    Reads multi-label ground truth exactly from the original CSV structures.
    """
    def __init__(self, frames_dir, labels_dir, transform=None):
        """
        frames_dir: Path to extracted frames, e.g., 'Galar_Frames_1_to_10 (1)'
        labels_dir: Path to the main directory containing 'Labels/' CSVs
        """
        self.frames_dir = frames_dir
        self.labels_dir = labels_dir
        self.transform = transform
        
        # Exact classes required by MTLEngine
        self.anatomical_classes = ["Stomach", "Small Bowel", "Colon"]
        self.lesion_classes = [
            "Normal", "Angioectasia", "Bleeding", "Chylous Cyst", 
            "Depressed", "Diverticulum", "Erosion", "Lymphangiectasia", 
            "Polyp", "Red Spots", "Ulcer", "Stricture"
        ]
        
        # Mappings from CSV column headers to the Engine index
        self.anatomy_map = {
            'stomach': 0,
            'small intestine': 1,
            'colon': 2
        }
        
        self.pathology_map = {
            'angiectasia': 1, # noting variation in dataset spelling if any
            'active bleeding': 2,
            'blood': 2, # map both to bleeding
            'diverticulum': 5, # if present in other csvs
            'erosion': 6,
            'lymphangioectasis': 7, # spelling match to csv
            'polyp': 8,
            'erythema': 9, # mapping red spots
            'ulcer': 10
        }

        self.samples = []
        self._load_data()

    def _load_data(self):
        """
        Scans the `Labels` directory for CSV files, pairs them with the
        corresponding folders in `frames_dir`, and builds the sample list.
        """
        labels_folder = os.path.join(self.labels_dir, 'Labels')
        if not os.path.exists(labels_folder):
            print(f"Error: Labels directory not found at {labels_folder}")
            return

        csv_files = glob.glob(os.path.join(labels_folder, '*.csv'))
        if not csv_files:
            print(f"Error: No CSV files found in {labels_folder}")
            return

        print(f"Found {len(csv_files)} annotation CSVs. Building dataset index...")
        
        for csv_path in csv_files:
            file_num = os.path.splitext(os.path.basename(csv_path))[0]
            img_folder_path = os.path.join(self.frames_dir, file_num)
            
            if not os.path.exists(img_folder_path):
                # We skip missing frame directories since the dataset might be partial
                continue
                
            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                print(f"Warning: Could not read {csv_path}: {e}")
                continue

            # Iterate through annotations
            for _, row in df.iterrows():
                frame_num = int(row['frame'])
                # The dataset names images like 'frame_000100.PNG'
                img_name = f'frame_{frame_num:06d}.PNG'
                img_path = os.path.join(img_folder_path, img_name)
                
                # Check if image actually exists (in cases of dropped frames during extraction)
                if not os.path.exists(img_path):
                    # Try JPG fallback just in case
                    img_path_jpg = os.path.join(img_folder_path, f'frame_{frame_num:06d}.jpg')
                    if os.path.exists(img_path_jpg):
                        img_path = img_path_jpg
                    else:
                        continue

                # Parse Anatomy (default to Small Bowel if undefined/ambiguous)
                anatomy_idx = 1 
                for col_name, idx in self.anatomy_map.items():
                    if col_name in df.columns and row[col_name] == 1:
                        anatomy_idx = idx
                        break
                        
                # Parse Pathology (Multi-label capable, returning a binary tensor)
                # But for the MTLEngine currently expecting single-class or BCE, we create a multi-hot array
                pathologies = np.zeros(len(self.lesion_classes), dtype=np.float32)
                is_normal = True
                
                for col_name, class_idx in self.pathology_map.items():
                    if col_name in df.columns and row.get(col_name, 0) == 1:
                        pathologies[class_idx] = 1.0
                        is_normal = False
                
                if is_normal:
                    pathologies[0] = 1.0 # Set "Normal" flag
                    
                self.samples.append({
                    'image_path': img_path,
                    'anatomy_idx': anatomy_idx,
                    'pathologies': pathologies # multi-hot encoded vector
                })

        print(f"Successfully indexed {len(self.samples)} physical frames matching CSV annotations.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = sample['image_path']
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception:
            # Fallback to zeros (black image) if physical file reading corrupts
            image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
            
        if self.transform:
            image = self.transform(image)
        else:
            # Basic fallback transform if none provided
            fallback_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
            image = fallback_transform(image)
            
        anatomy_label = torch.tensor(sample['anatomy_idx'], dtype=torch.long)
        pathology_label = torch.tensor(sample['pathologies'], dtype=torch.float32)
        
        return image, anatomy_label, pathology_label

