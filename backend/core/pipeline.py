import os
import cv2
import time
from backend.utils.preprocessing import Preprocessor
from backend.models.quality_gate import QualityGate
from backend.models.mtl_engine import MTLEngine
from backend.models.temporal import TemporalRefiner
from backend.models.xai import XAIGenerator

class DeepCarePipeline:
    def __init__(self):
        self.preprocessor = Preprocessor()
        self.quality_gate = QualityGate()
        self.mtl_engine = MTLEngine()
        self.temporal_refiner = TemporalRefiner()
        self.xai = XAIGenerator()

    def process_video(self, video_path, output_dir):
        """
        Runs the full 5-step Deep Care pipeline on a video.
        """
        results = []
        frames_generator = self.preprocessor.extract_frames(video_path, sample_rate=5) # Process every 5th frame for speed
        
        processed_dir = os.path.join(output_dir, "processed")
        os.makedirs(processed_dir, exist_ok=True)

        for frame_idx, frame in frames_generator:
            # Step 1: Pre-processing
            enhanced_frame = self.preprocessor.preprocess_frame(frame)
            
            # Step 2: Quality Gate
            is_good, q_conf = self.quality_gate.is_informative(enhanced_frame)
            
            if not is_good:
                results.append({
                    "frame_idx": frame_idx,
                    "status": "skipped",
                    "reason": "low_quality",
                    "quality_score": float(q_conf)
                })
                continue

            # Step 3: MTL Engine
            prediction = self.mtl_engine.predict(enhanced_frame)
            
            # Step 4: Temporal Consistency
            refined_pred = self.temporal_refiner.smooth_prediction(prediction)
            
            # Step 5: XAI
            # Only generate heatmap for abnormalities to save space/time
            heatmap_path = None
            if refined_pred['pathology'] != 'Normal':
                # We need to access the internal tensor from the prediction for XAI
                # This is a bit tighter coupling but efficient
                input_tensor = prediction.get('input_tensor')
                
                if input_tensor is not None:
                    heatmap_raw = self.xai.generate_heatmap(
                        self.mtl_engine.get_model(), 
                        input_tensor
                    )
                    
                    if heatmap_raw is not None:
                        overlay, _ = self.xai.overlay_heatmap(enhanced_frame, heatmap_raw)
                        heatmap_filename = f"heatmap_{frame_idx}.jpg"
                        heatmap_full_path = os.path.join(processed_dir, heatmap_filename)
                        cv2.imwrite(heatmap_full_path, overlay)
                        heatmap_path = f"processed/{heatmap_filename}"
            
            # Clean up tensor to be safe for JSON serialization
            if 'input_tensor' in prediction:
                del prediction['input_tensor']
                
            if 'input_tensor' in refined_pred:
                del refined_pred['input_tensor']
            
            # Save original enhanced frame reference? 
            # For now just save the metadata
            
            results.append({
                "frame_idx": frame_idx,
                "status": "processed",
                "quality_score": float(q_conf),
                "prediction": refined_pred,
                "heatmap_url": heatmap_path
            })
            
        return results
