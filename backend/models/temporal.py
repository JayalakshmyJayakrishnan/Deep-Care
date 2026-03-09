from collections import deque
import numpy as np

class TemporalRefiner:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.history = deque(maxlen=window_size)
    
    def smooth_prediction(self, current_pred):
        """
        Phase 3: Temporal Refinement.
        Uses sliding window to smooth predictions.
        """
        self.history.append(current_pred)
        
        # Simple majority voting for pathology
        pathologies = [p['pathology'] for p in self.history]
        if not pathologies:
            return current_pred
            
        # Find most common
        most_common = max(set(pathologies), key=pathologies.count)
        
        # If the most common is different from current, and current has low confidence,
        # we might swap it (Simplified Viterbi-like logic)
        
        refined_pred = current_pred.copy()
        refined_pred['original_pathology'] = current_pred['pathology']
        refined_pred['pathology'] = most_common
        refined_pred['is_smoothed'] = (most_common != current_pred['pathology'])
        
        return refined_pred
