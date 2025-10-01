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

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(image: UploadFile = File(...)):
    try:
        ## -- STUDENT CODE HERE --
        print("Received file:", image.filename)
        ## -- STUDENT CODE END -- 
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


# To run the app from root directory: uvicorn app.api:app --reload
# Test by going to 127.0.0.1:8000/docs in your browser
    
