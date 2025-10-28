import io
import cv2
import uvicorn
import numpy as np
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from .model import InferenceModel

app = FastAPI(title="Edge CV API")

MODEL_WEIGHTS = os.path.join(os.path.dirname(__file__), '..', 'runs', 'detect', 'train', 'weights', 'best.pt') # edit `train` if you are using train2 etc
model = InferenceModel(weights_path=MODEL_WEIGHTS)


try:
    model = InferenceModel(weights_path=MODEL_WEIGHTS, device="cpu") 
except Exception as e:
    print(f"Failed to load model weights.")
    model = None

app = FastAPI(title="Edge CV API")

@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    return {"status": "ok"}

@app.post("/predict")
def predict(image: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    try:
        contents = image.file.read() 
        np_arr = np.frombuffer(contents, np.uint8)

        image_bgr = cv2.imdeconde(np_arr, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise HTTPException(status_code=422, detail="Could not decode image.")
        
        detections = model.predict(image_bgr=image_bgr)
        return {"filename": image.filename, "detections": detections}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


# To run the app from root directory: uvicorn app.api:app --reload
# Test by going to 127.0.0.1:8000/docs in your browser
    
