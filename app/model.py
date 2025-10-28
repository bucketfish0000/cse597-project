# app/model.py
from typing import List, Dict, Any
from ultralytics import YOLO
from PIL import Image
import torch
import numpy as np
import os

class ModelLoadError(Exception):
    pass

class InferenceError(Exception):
    pass

class InferenceModel:
    def __init__(self, weights_path: str, device: str = "cpu"):
        """Load weights and initialize runtime.
        """
        if not os.path.exists(weights_path):
            raise ModelLoadError(f"Weight file not existent: {weights_path}")
        
        if device.startswith('cuda') and not torch.cuda.is_available():
            print(f"Device {device} not available, using CPU by default")
            self.device = 'cpu'
        else:
            self.device = device

        try:
            self.model = YOLO(weights_path)
            self.model.to(self.device)
        except Exception as e:
            raise ModelLoadError(f"Failed to load model: {e}")

    def predict(self, image_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """Return a list of detections.
        For detection: [{"label": str, "score": float, "bbox": [x1,y1,x2,y2]}]
        """
        if not isinstance(image_bgr, np.ndarray) or image_bgr.ndim != 3 or image_bgr.shape[2] not in [1, 3, 4]:
            raise InferenceError(f"Incompatible image input format: {image_bgr.shape}")
        
        try:
            results_list = self.model.predict(
                source=image_bgr, 
                device=self.device, 
                conf=0.25,
                iou=0.7,  
                verbose=False 
            )
        except Exception as e:
            raise InferenceError(f"Error in inference: {e}")   
        
        detections: List[Dict[str, Any]] = []

        if results_list:
            result = results_list[0]
            if result.boxes and result.boxes.data.numel() > 0:
                boxes_data = result.boxes.data.cpu().numpy()
                
                names = result.names 

                for box in boxes_data:
                    x1, y1, x2, y2, score, class_id = box
                    class_id = int(class_id)
                    
                    detection = {
                        "label": names.get(class_id, "unknown"), 
                        "score": float(score),
                        "bbox": [int(x1), int(y1), int(x2), int(y2)] 
                    }
                    detections.append(detection)
                    
        return detections