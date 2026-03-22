import os
import sys
import argparse
import random
import glob
import json
import uuid
from datetime import datetime
import cv2

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.core.pipeline import DeepCarePipeline

def evaluate_dataset(dataset_dir, num_samples, output_dir):
    print(f"--- DeepCare Pipeline: Hospital Dataset Evaluation ---")
    print(f"Target Dataset: {dataset_dir}")
    print(f"Sampling: {num_samples} frames")
    
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory {dataset_dir} not found.")
        sys.exit(1)
        
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Collect all frames
    print("Indexing dataset frames...")
    all_frames = []
    extensions = ['*.jpg', '*.png']
    for ext in extensions:
        all_frames.extend(glob.glob(os.path.join(dataset_dir, '**', ext), recursive=True))
        
    if not all_frames:
        print("Error: No frames found in the specified dataset directory.")
        sys.exit(1)
        
    print(f"Found {len(all_frames)} total frames.")
    
    # 2. Sample frames
    sample_size = min(num_samples, len(all_frames))
    sampled_paths = random.sample(all_frames, sample_size)
    
    print(f"Selected {sample_size} random frames for evaluation.")
    
    # Initialize pipeline
    print("Initializing DeepCare Pipeline...")
    pipeline = DeepCarePipeline()
    
    study_id = f"EVAL-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    eval_output_dir = os.path.join(output_dir, study_id)
    os.makedirs(eval_output_dir, exist_ok=True)
    
    report = {
        "study_id": study_id,
        "timestamp": datetime.now().isoformat(),
        "clinical_context": {
            "department": "Gastroenterology",
            "report_type": "Automated AI PillCam Diagnostics",
            "dataset_reference": os.path.basename(dataset_dir)
        },
        "total_frames_sampled": sample_size,
        "metrics": {
            "processed": 0,
            "skipped_low_quality": 0,
            "abnormalities_detected": 0,
            "anatomy_breakdown": {},
            "pathology_breakdown": {}
        },
        "findings": []
    }
    
    print("\n--- Starting Processing ---")
    
    # Process each frame independently
    # The pipeline is typically built for video iteration, but we can simulate a video
    # Since TemporalRefiner requires sequential frames for smoothing, processing random 
    # independent frames will just use the raw MTL prediction for each, which is fine for bulk analysis.
    
    # To use the exact pipeline method `process_video`, we would need a video file.
    # Instead, we will wrap the frames in a mock generator or bypass `process_video` and call the components.
    
    # Bypassing `process_video` wrapper to tightly control individual frame processing and reporting.
    
    for i, frame_path in enumerate(sampled_paths):
        print(f"[{i+1}/{sample_size}] Processing: {os.path.basename(frame_path)}")
        try:
            # Load frame
            frame = cv2.imread(frame_path)
            if frame is None:
                print(f"  -> Failed to read image")
                continue
                
            # Step 1: Pre-processing
            enhanced_frame = pipeline.preprocessor.preprocess_frame(frame)
            
            # Step 2: Quality Gate
            is_good, q_conf = pipeline.quality_gate.is_informative(enhanced_frame)
            
            if not is_good:
                print(f"  -> Skipped: Low Quality ({q_conf:.2f})")
                report["metrics"]["skipped_low_quality"] += 1
                continue
                
            # Step 3: MTL Engine
            prediction = pipeline.mtl_engine.predict(enhanced_frame)
            
            # Since these are random non-sequential frames, we skip the Temporal Refiner
            refined_pred = prediction # Raw is fine for non-temporal random sampling
            
            anatomy = refined_pred.get('anatomical', 'Unknown')
            pathology = refined_pred.get('pathology', 'Unknown')
            path_conf = refined_pred.get('pathology_confidence', 0.0)
            
            # Update metrics
            report["metrics"]["processed"] += 1
            
            report["metrics"]["anatomy_breakdown"][anatomy] = report["metrics"]["anatomy_breakdown"].get(anatomy, 0) + 1
            report["metrics"]["pathology_breakdown"][pathology] = report["metrics"]["pathology_breakdown"].get(pathology, 0) + 1
            
            is_abnormal = pathology != 'Normal'
            if is_abnormal:
                report["metrics"]["abnormalities_detected"] += 1
                
            # Step 5: XAI (Only for abnormalities)
            heatmap_saved_path = None
            if is_abnormal:
                input_tensor = refined_pred.get('input_tensor')
                if input_tensor is not None:
                    heatmap_raw = pipeline.xai.generate_heatmap(
                        pipeline.mtl_engine.get_model(), 
                        input_tensor
                    )
                    
                    if heatmap_raw is not None:
                        overlay, _ = pipeline.xai.overlay_heatmap(enhanced_frame, heatmap_raw)
                        heatmap_filename = f"heatmap_{uuid.uuid4().hex[:8]}.jpg"
                        heatmap_full_path = os.path.join(eval_output_dir, heatmap_filename)
                        cv2.imwrite(heatmap_full_path, overlay)
                        heatmap_saved_path = os.path.relpath(heatmap_full_path, output_dir)
            
            print(f"  -> Anatomy: {anatomy} | Pathology: {pathology} ({path_conf:.2f})")
            
            finding = {
                "frame_filename": os.path.basename(frame_path),
                "quality_score": round(float(q_conf), 4),
                "anatomy": anatomy,
                "pathology": pathology,
                "confidence": round(float(path_conf), 4),
                "heatmap_url": heatmap_saved_path
            }
            report["findings"].append(finding)
            
        except Exception as e:
            print(f"  -> Error processing frame: {e}")
            
            
    # Save Report
    report_path = os.path.join(eval_output_dir, 'hospital_evaluation_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
        
    print("\n--- Evaluation Complete ---")
    print(f"Processed: {report['metrics']['processed']}")
    print(f"Skipped (Low Quality): {report['metrics']['skipped_low_quality']}")
    print(f"Abnormalities Detected: {report['metrics']['abnormalities_detected']}")
    
    print("\nPathology Breakdown:")
    for path, count in report["metrics"]["pathology_breakdown"].items():
        print(f"  - {path}: {count}")
        
    print(f"\nDetailed report and heatmaps saved to: {eval_output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate DeepCare Pipeline on Dataset')
    parser.add_argument('--dataset', type=str, default=r'd:\MINOR\pillcam_platform\datasets\Galar_Frames_1_to_10 (1)', help='Path to dataset frames')
    parser.add_argument('--samples', type=int, default=50, help='Number of frames to sample and evaluate')
    parser.add_argument('--output', type=str, default=r'd:\MINOR\pillcam_platform\backend\eval_results', help='Output directory for report and heatmaps')
    
    args = parser.parse_args()
    evaluate_dataset(args.dataset, args.samples, args.output)
