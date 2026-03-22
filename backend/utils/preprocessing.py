import cv2
import numpy as np

class Preprocessor:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        # Create CLAHE object (Arguments are optional)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    def apply_clahe(self, image):
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to normalize lighting
        and enhance mucosal textures.
        """
        if image is None:
            return None
            
        # CLAHE is applied on the L channel of LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        l_clahe = self.clahe.apply(l)
        
        lab_merged = cv2.merge((l_clahe, a, b))
        return cv2.cvtColor(lab_merged, cv2.COLOR_LAB2BGR)

    def preprocess_frame(self, frame):
        """
        Resize and enhance frame.
        """
        if frame is None:
            return None
        
        # Resize to standardized 224x224
        resized = cv2.resize(frame, self.target_size)
        
        # Apply CLAHE
        enhanced = self.apply_clahe(resized)
        
        return enhanced

    def extract_frames(self, video_path, sample_rate=1):
        """
        Generator to extract frames from video or a single frame from an image.
        sample_rate: Process every Nth frame.
        """
        import os
        ext = os.path.splitext(video_path)[1].lower()
        if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            frame = cv2.imread(video_path)
            if frame is not None:
                yield 0, frame
            return

        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % sample_rate == 0:
                yield frame_count, frame
                
            frame_count += 1
            
        cap.release()
