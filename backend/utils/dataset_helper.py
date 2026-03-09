import os
import random
import glob

class DatasetHelper:
    def __init__(self, dataset_name='galar'):
        # Map generic 'galar' request to the major dataset folder
        if dataset_name == 'galar':
            dataset_name = os.path.join('galar', 'Galar_Frames_1_to_10 (1)')
            
        self.base_dir = os.path.join(os.getcwd(), 'datasets', dataset_name)
        
    def get_random_sample(self):
        """
        Returns the path to a random image/video file from the dataset.
        """
        if not os.path.exists(self.base_dir):
            return None
            
        # Recursive search for media files
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.mp4', '*.avi']
        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(self.base_dir, '**', ext), recursive=True))
            
        if not files:
            return None
            
        return random.choice(files)
