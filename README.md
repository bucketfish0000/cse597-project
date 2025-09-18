# Computer Vision Assignment: Serve a Model via Local API + Raspberry Pi Client


**Estimated effort:** 5-7 hours

---

## Overview

You’ll build an end‑to‑end, industry‑style inference pipeline:

1. **Train** a computer vision model.
2. **Package it in** with a clean, testable interface.
3. **Expose a local HTTP API (**``**)** so any device (a Raspberry Pi robot) can request predictions.
4. **Write a Raspberry Pi client** that captures images and posts them to your API.

This mirrors how models are deployed in practice: the robot sends frames to a nearby machine hosting the model service (“on‑the‑edge” inference).

---

## Learning Objectives

By the end you can:

- Implement a **stable inference API** around a computer vision model.
- Serve predictions with a **FastAPI** endpoint and validate inputs/outputs.
- **Stream images** from a Raspberry Pi to an API, and handle errors/latency.

---

## Prerequisites & Materials

- Python 3.10+; Conda or venv.
- **FastAPI** + `uvicorn`, `pydantic`, `opencv-python`, `numpy`, `Pillow`, `requests`.
- **Picar-X** (given in class)

> Do not expect to have access to a GPU due to limited resources

---

## Repository Layout 

```
cse597-project/
├── .gitignore
├── README.md
├── Getting Started.pdf
├── requirements.txt
├── client.py         # /!\ MODIFY /!\ Part 4
├── venv/               
├── app/
│   ├── api.py        # /!\ MODIFY /!\ Part 3
│   └── model.py      # /!\ MODIFY /!\ Part 2
├── data/             # add this folder from unzipping dataset
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

## Part 0 — Environment & Setup (5 pts)

- Read through and follow the entire Getting Started PDF
- Python, VSCode, a virtual environment, and git should be set up and the dataset and the project repository should be downloaded onto your local computer
- Points will be awarded if any parts of the project are submitted 

## Part 1 — Model Training (20 pts)

 In `notebooks/train.ipynb`, fine‑tune a compact model on the `object_detection` dataset provided with the assignment. 

---

## Part 2 — `model.py`: Inference Contract (25 pts)

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
        Coordinates are in **original image pixels**.
        """
        ...
```

**Requirements:**

- **Deterministic** output for a given image.
- Graceful errors with helpful messages (bad image, wrong shape, missing weights).
- Answer questions 
---

## Part 3 — `api.py`: FastAPI Service (25 pts)

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
from fastapi import FastAPI, File, UploadFile, HTTPException
from .model import InferenceModel

app = FastAPI(title="Edge CV API")
model = InferenceModel("models/model.pt", device="cpu")

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


---

## Part 4 — Raspberry Pi Client (25 pts)

Write `client/pi_client.py` that captures frames and POSTs to the API.

**Skeleton:**

```python
# client/pi_client.py
import time
import cv2
import requests

API_URL = "http://<laptop-ip>:8000/predict"  # replace with server IP
CAM_INDEX = 0

cap = cv2.VideoCapture(CAM_INDEX)
assert cap.isOpened(), "Camera not found"

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        _, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        files = {"image": ("frame.jpg", buf.tobytes(), "image/jpeg")}
        t0 = time.perf_counter()
        r = requests.post(API_URL, files=files, timeout=5)
        dt = (time.perf_counter() - t0) * 1000
        r.raise_for_status()
        preds = r.json()
        print(f"Latency: {dt:.1f} ms | preds: {preds}")
        # OPTIONAL: draw boxes and show window
        # for det in preds: ...
        # cv2.imshow("pi", frame); cv2.waitKey(1)
except KeyboardInterrupt:
    pass
finally:
    cap.release()
```


---


**Breakdown:**

- Part 0 (5) + Part 1 (20) + Part 2 (25) + Part 3 (25) + Part 4 (25)  = **100 pts**

---

**Deliverables:** code + short write‑up (1–2 paragraphs) + demo in class


## Suggested Requirements File

```
fastapi
uvicorn
pydantic
numpy
opencv-python
pillow
requests
# choose one stack
# torch
torch
# OR onnxruntime
# onnxruntime
# OR tflite-runtime (on Pi)
# tflite-runtime
pytest
ruff
```

