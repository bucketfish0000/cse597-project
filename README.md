# Homework 4: Computer Vision Assignment: Serve a Model via Local API 


**Estimated effort:** 5-7 hours

---

## Overview

You’ll build an end‑to‑end, industry‑style inference pipeline:

1. **Train** a computer vision model.
2. **Package it in** with a clean, testable interface.
3. **Expose a local HTTP API (**``**)** so any device (a Raspberry Pi robot) can request predictions.

This mirrors how models are deployed in practice: the robot sends frames to a nearby machine hosting the model service (“on‑the‑edge” inference).

---

## Learning Objectives

By the end you can:

- **Train a YOLO model** on a simple dataset
- Implement a **stable inference API** around a custom YOLO model.
- Serve predictions with a **FastAPI** endpoint and validate inputs/outputs.

---

## Repository Layout 

```
cse597-project/
├── .gitignore
├── README.md
├── Getting Started.pdf
├── requirements.txt
├── venv/               
├── app/
│   ├── report.md     # /!\ MODIFY /!\ Part 2 & 3
│   ├── model.py      # /!\ MODIFY /!\ Part 2
│   └── api.py        # /!\ MODIFY /!\ Part 3
├── data/             # dataset unzipped here 
│   ├── README.roboflow.txt
│   ├── data.yaml
│   ├── train/
│   │   ├── images/
│   │   ├── labels/
│   │   └── labels.cache
│   ├── valid/
│   │   ├── images/
│   │   ├── labels/
│   │   └── labels.cache
│   └── test/
│       ├── images/
│       └── labels/
├── notebooks/
│   ├── train.ipynb # /!\ MODIFY /!\ Part 1
│   └── yolov8n.pt
├── runs/           # this folder is created from running train.ipynb
│   └── detect/
│       ├── train/
│       │   ├── weights/
│       │   │   ├── best.pt
│       │   │   └── last.pt
│       │   ├── BoxF1_curve.png
│       │   ├── BoxPR_curve.png
│       │   ├── BoxP_curve.png
│       │   ├── BoxR_curve.png
│       │   ├── args.yaml
│       │   ├── confusion_matrix.png
│       │   ├── confusion_matrix_normalized.png
│       │   ├── labels.jpg
│       │   ├── results.csv
│       │   ├── results.png
│       │   ├── train_batch0.jpg
│       │   ├── train_batch1.jpg
│       │   ├── train_batch2.jpg
│       │   ├── val_batch0_labels.jpg
│       │   ├── val_batch0_pred.jpg
│       │   ├── val_batch1_labels.jpg
│       │   ├── val_batch1_pred.jpg
│       │   ├── val_batch2_labels.jpg
│       └──  └── val_batch2_pred.jpg
└── .vscode/
```

---

## Part 0 — Environment & Setup (10 pts)

- Read through and follow the entire Getting Started PDF
- Python, VSCode, a virtual environment, and git should be set up and the dataset and the project repository should be downloaded onto your local computer
- Points will be awarded if any parts of the project are submitted 
- Contact Sabrina with questions

## Part 1 — Model Training (30 pts)

 In `notebooks/train.ipynb`, fine‑tune a YOLO model on the `object_detection` dataset provided with the assignment. 


 Additionally, there are questions about the dataset and about the model training that you need to answer.

---

## Part 2 — `model.py`: Inference Wrapper (30 pts)

Create an interface wrapper around your model so the API and tests don’t depend on framework internals.

**Must Implement:**

```python
# app/model.py
from typing import List, Dict, Any
import numpy as np

class InferenceModel:
    def __init__(self, weights_path: str, device: str = "cpu"):
        """Load weights and initialize runtime.
        """
        ...

    def predict(self, image_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """Return a list of detections or classifications.
        For detection: [{"label": str, "score": float, "bbox": [x1,y1,x2,y2]}]
        """
        ...
```

**Requirements:**

- **Deterministic** output for a given image.
- Graceful errors with helpful messages (bad image, wrong shape, missing weights).
- Answer questions in `report.md`
---

## Part 3 — `api.py`: FastAPI Service (30 pts)

Implement a local HTTP service that wraps `InferenceModel`.

**Endpoints (minimum):**

- `GET /health` → `{ "status": "ok" }`
- `POST /predict` (multipart form, key=`image`): returns prediction.

**Skeleton:**

```python
# app/api.py
import io
import cv2
import uvicorn
import numpy as np
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from .model import InferenceModel

app = FastAPI(title="Edge CV API")

MODEL_WEIGHTS = os.path.join(os.path.dirname(__file__), '..', 'runs', 'detect', 'train', 'weights', 'best.pt')
model = InferenceModel(weights_path=MODEL_WEIGHTS)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(image: UploadFile = File(...)):
    try:
        ## -- STUDENT CODE HERE --

        ## -- STUDENT CODE END -- 
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Run:** `uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload`

**Requirements:**

- Be able to give a demo of the server hosted locally 
- Handle errors 
- Answer questions in `report.md`

---

**Breakdown:**

- Part 0 (10) + Part 1 (30) + Part 2 (30) + Part 3 (30)  = **100 pts**

---

**Deliverables:** train.ipynb & write ups in this notebook + api.py + model.py + report.md 

*This needs to be completed in a thorough way for the next section of the course. Please contact **Sabrina** with questions and concerns*


