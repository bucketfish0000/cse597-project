# app/model.py
from typing import List, Dict, Any
import numpy as np

class InferenceModel:
    def __init__(self, weights_path: str, device: str = "cpu"):
        """Load weights and initialize runtime.
        Must set self.input_size (w, h) and self.labels (list or dict).
        """
        ...

    def preprocess(self, image_bgr: np.ndarray) -> Any:
        """Resize/normalize and convert to the model's input tensor."""
        ...

    def predict(self, image_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """Return a list of detections or classifications.
        For detection: [{"label": str, "score": float, "bbox": [x1,y1,x2,y2]}]
        For classification: [{"label": str, "score": float}]
        Coordinates are in **original image pixels**.
        """