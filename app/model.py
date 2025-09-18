# app/model.py
from typing import List, Dict, Any
import numpy as np

class InferenceModel:
    def __init__(self, weights_path: str, device: str = "cpu"):
        """Load weights and initialize runtime.
        """
        ...

    def predict(self, image_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """Return a list of detections.
        For detection: [{"label": str, "score": float, "bbox": [x1,y1,x2,y2]}]
        """