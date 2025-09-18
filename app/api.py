from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

import model  # Assumes model.py is in the same directory

app = FastAPI()

class PredictionRequest(BaseModel):
    data: list  # Adjust type as needed for your model

class PredictionResponse(BaseModel):
    prediction: list  # Adjust type as needed for your model

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # -- student add code here --
        print("Add code here")
        # -- end student code --
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
